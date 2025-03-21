# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for IDEFICS.
"""
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from typing import Callable, List, Optional, Union
from urllib.parse import urlparse

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType, is_torch_available

from typing import Optional

import torch
from loguru import logger
from transformers import BatchFeature, IdeficsForVisionText2Text


if is_torch_available():
    import torch


IMAGE_TOKEN = "<image>"

from io import BytesIO
from typing import List

import requests
import torch
from loguru import logger
from PIL import Image
from torch import nn
from lmm_icl_interface.utils import cast_type, get_autocast, is_url
from lmm_icl_interface.prompt_control import LMMPromptManager


class BaseInterface(nn.Module):
    def __init__(
        self,
        precision,
        input_ids_field_name: str,
        prompt_manager: LMMPromptManager,
        instruction: str,
        label_field: str,
    ) -> None:
        super().__init__()
        self.data_type = cast_type(precision)
        self.autocast_context = get_autocast(precision)
        self.input_ids_field_name = input_ids_field_name

        self.prompt_manager = prompt_manager
        self.instruction = instruction
        self.pad_token_id = None
        self.tokenizer = None
        self.label_field = label_field

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        else:
            logger.warning("the model has not device parameters")
            return None

    @torch.inference_mode()
    def get_cond_prob(
        self,
        model_input,
        mask_length=None,
    ):
        ce_loss = self.get_ppl(model_input, mask_length)
        return (-ce_loss).exp()

    @torch.inference_mode()
    def get_ppl(
        self,
        model_input,
        mask_length=None,
    ):
        if self.pad_token_id is None:
            logger.warning("the pad_token_id is None")
        with self.autocast_context:
            outputs = self.model(**model_input)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = model_input[self.input_ids_field_name][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=self.pad_token_id
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())

            if mask_length is not None:
                loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
                for i in range(len(loss_mask)):
                    for j in range(mask_length[i] - 1, len(loss_mask[i])):
                        loss_mask[i][j] = 1
                loss = loss * loss_mask
            lens = (model_input[self.input_ids_field_name] != self.pad_token_id).sum(-1)

            if mask_length is not None:
                lens -= torch.tensor(mask_length, device=lens.device)

            ce_loss = loss.sum(-1) / lens
        return ce_loss

    def transfer_icl_prompts(
        self, batch_data_sample_list, is_last_for_generation=True, query_label=None
    ):
        """
        transfer data sample list to text input.
        Note: Only support one image and one text pair.
        """
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.extend(
                    [
                        data_sample[self.image_field],
                        self.prompt_manager.gen_ice_text_with_label(
                            data_sample, add_sep_token=True
                        ),
                    ]
                )
            prompt.append(data_sample_list[-1][self.image_field])
            if is_last_for_generation:
                prompt.append(
                    self.prompt_manager.gen_query_text_without_label(
                        data_sample_list[-1]
                    )
                )
            else:
                prompt.append(
                    self.prompt_manager.gen_query_text_with_label(
                        data_sample_list[-1], label=query_label
                    )
                )

            prompts.append(prompt)
        return prompts

    def generate(self, *args, **kwargs):
        with self.autocast_context:
            return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with self.autocast_context:
            return self.model(*args, **kwargs)


class LMMInterface(BaseInterface):
    def __init__(
        self,
        precision,
        input_ids_field_name,
        prompt_manager: LMMPromptManager,
        instruction,
        label_field,
        image_field,
    ):
        super().__init__(
            precision=precision,
            input_ids_field_name=input_ids_field_name,
            prompt_manager=prompt_manager,
            instruction=instruction,
            label_field=label_field,
        )

        self.image_field = image_field

    def is_img(self, obj):
        if isinstance(obj, Image.Image):
            return obj
        elif isinstance(obj, str):
            if is_url(obj):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                        " Safari/537.36"
                    )
                }
                response = requests.get(obj, stream=True, headers=headers)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                try:
                    return Image.open(obj)
                except:
                    return None

# copied from m4.training.packing
def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    # This function converts: [-1, 0, 1] => [[0, 0], [1, 0], [0, 1]]

    # If any of images index are more than num_classes, set them to -1.
    # Words after the max number of images allowed have been seen don't attend on anything
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1

    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask


# copied from m4.training.packing
def image_attention_mask_for_packed_input_ids(input_ids, tokenizer):
    image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    next_image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    eod_token_id = tokenizer.eos_token_id
    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx, token_id in enumerate(input_ids[batch_idx]):
            if token_id == image_token_id:
                count += 1
                image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                image_attention_mask[batch_idx][idx] = count

            if seen_eod:
                image_attention_mask[batch_idx][idx] = -1

            if token_id == eod_token_id:
                seen_eod = True

    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx in range(input_ids[batch_idx].size(0) - 1, -1, -1):
            token_id = input_ids[batch_idx][idx]
            if token_id == image_token_id:
                count += 1
                next_image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                next_image_attention_mask[batch_idx][idx] = count

            if token_id == eod_token_id:
                seen_eod = True

            if seen_eod:
                next_image_attention_mask[batch_idx][idx] = -1

        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1

    return image_attention_mask, next_image_attention_mask


def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


class IdeficsProcessor(ProcessorMixin):
    r"""
    Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`IdeficsImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`IdeficsImageProcessor`):
            An instance of [`IdeficsImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224): Image size (assuming a square image)
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "IdeficsImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        self.default_image_dims = (
            self.image_processor.image_num_channels,
            self.image_processor.image_size,
            self.image_processor.image_size,
        )

        self.tokenizer_was_trained_with_end_of_utterance_token = (
            True
            if "<end_of_utterance>" in self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            else False
        )

    def __call__(
        self,
        prompts: Union[List[TextInput], List[List[TextInput]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        transform: Callable = None,
        add_eos_token=False,
        add_end_of_utterance_token=None,
        debug=False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchEncoding:
        """This method takes batched or non-batched prompts made of text and images and converts them into prompts that
        the model was trained on and prepares the image pixel values for the model to process.

        Args:
            prompts (`Union[List[TextInput], [List[List[TextInput]]]]`):
                either a single prompt or a batched list of prompts - see the detailed description immediately after
                the end of the arguments doc section.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            transform (`Callable`, *optional*):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple functions. If `None` a preset inference-specific
                set of transforms will be applied to the images
            add_eos_token (`bool`, *optional*, defaults to `False`):
                Adds `eos_token` at the end of the final prompt if True`
            add_end_of_utterance_token (`bool`, *optional*)
                Whether to automatically add `<end_of_utterance>` after each prompt's text input (unless followed by an
                image). If `None` the tokenizer will be checked instead and if this token is found in
                `additional_special_tokens` then the value will be `True`.
            debug (`bool`, *optional*, defaults to `False`):
                `True` value will help debug prompt generation by dumping useful information
            return_tensors (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`):
                The type of tensors to return. Can be one of:
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.

        Returns:
            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
            directly passed to `model.generate`

        Detailed explanation:

        Each entry in `prompts` is either a text to be passed as is or an image that will be processed.

        An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

        When the processor encounters an image it'll inject `<fake_token_around_image><image><fake_token_around_image>`
        entry into the prompt.

        Example:

        ```python
        checkpoint = "HuggingFaceM4/idefics-9b"
        processor = AutoProcessor.from_pretrained(checkpoint)
        url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
        img = processor.image_processor.fetch_images([url])[0]

        prompts = [
            "User:",
            img,
            "Describe this image.\nAssistant: An image of two kittens in grass.\n",
            "User:",
            "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
            "Describe this image.\nAssistant:",
        ]

        inputs = processor(prompts, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```

        In this example the `prompts` will be converted into:

        ```
        <s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant: An image of two kittens in grass.
        User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant:'
        ```

        and the two images will be massaged using [`IdeficsImageProcessor.__call__`] method and placed inside the
        `pixel_values` dict entry of the return value.

        This example also examplifies that images can be passed as objects or as text urls. It can be seen that the
        first image is passed as object and the second one as a url.

        To do training do:

        ```python
        image_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (w, h), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        inputs = processor(prompts, transform=image_transform, return_tensors="pt")
        ```

        In order to help debug prompt generation enable `debug=True` which will show you what's happening.

        """

        # if the value isn't overriden by the user, check if the tokenizer was trained with this token and then use it
        if add_end_of_utterance_token is None:
            add_end_of_utterance_token = self.tokenizer_was_trained_with_end_of_utterance_token

        # turn non-batched prompts into batched
        if not any(isinstance(i, list) for i in prompts):
            prompts = [prompts]

        fake_token = "<fake_token_around_image>"
        image_token = "<image>"
        end_of_utterance_token = "<end_of_utterance>"

        def image_tokens(last_was_image):
            if last_was_image:
                return image_token + fake_token
            else:
                return fake_token + image_token + fake_token

        all_prompts = []
        all_images = []
        for sample in prompts:
            # the model was trained on samples starting with <s>
            full_text = f"{self.tokenizer.bos_token}"

            # an image can either be an image object in the item or the url, everything else is a verbatim prompt text
            image_objects = []
            last_was_image = False
            last_was_text = False
            for i, item in enumerate(sample):
                if i > 0:
                    last_was_text = True if not last_was_image else False

                if isinstance(item, str):
                    item = item.strip(" ")
                    if is_url(item):
                        image = self.image_processor.fetch_images(item)
                        full_text += image_tokens(last_was_image)
                        image_objects.append(image)
                        last_was_image = True
                    else:
                        # we add end_of_utterance_token between each subsequent text prompts (but not at the last one!)
                        if add_end_of_utterance_token and last_was_text:
                            full_text += end_of_utterance_token
                        full_text += item
                        last_was_image = False
                else:
                    # must be an image obj
                    full_text += image_tokens(last_was_image)
                    image_objects.append(item)
                    last_was_image = True

            if add_eos_token:
                full_text += self.tokenizer.eos_token

            if debug is True:
                print(f"{full_text=}")

            image_objects = self.image_processor(image_objects, transform=transform)

            all_prompts.append(full_text)
            all_images.append(image_objects)

        text_encoding = self.tokenizer(
            text=all_prompts,
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        all_texts = text_encoding["input_ids"]

        max_seq_len = max(len(x) for x in all_texts)

        # max_num_images has to be at least 1 even when there are no images
        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        at_least_one_image = sum(len(x) for x in all_images) > 0
        output_input_ids = []
        output_images = []
        output_attention_masks = []
        for text, images in zip(all_texts, all_images):
            padded_input_ids = [self.tokenizer.pad_token_id] * max_seq_len
            unpadded_seq_len = len(text)
            start = max_seq_len - unpadded_seq_len
            padded_input_ids[start:] = text[:max_seq_len]

            attention_mask = torch.zeros((max_seq_len,), dtype=torch.long)
            attention_mask[start:] = 1

            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, max_num_images)

            current_images = images[:local_max_num_images]

            if len(current_images) > 0:
                padded_image_tensor = torch.zeros(max_num_images, *current_images.size()[1:])
                padded_image_tensor[: current_images.size(0)] = current_images
            else:
                padded_image_tensor = torch.zeros(max_num_images, *self.default_image_dims)

            output_images.append(padded_image_tensor)
            output_input_ids.append(torch.tensor(padded_input_ids))

            output_attention_masks.append(attention_mask)

        output_input_ids = torch.stack(output_input_ids)
        output_images = torch.stack(output_images)
        output_attention_masks = torch.stack(output_attention_masks)

        if at_least_one_image:
            image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, self.tokenizer)
            image_attention_mask = incremental_to_binary_attention_mask(
                image_attention_mask, num_classes=max_num_images
            )
        else:
            # in full language mode we set the image mask to all-0s
            image_attention_mask = torch.zeros(
                output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool
            )

        return BatchFeature(
            data={
                "input_ids": output_input_ids,
                "attention_mask": output_attention_masks,
                "pixel_values": output_images,
                "image_attention_mask": image_attention_mask,
            }
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))




def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


class LMMPromptProcessor:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.input_ids_field = "input_ids"

    def set_input_ids_field(self, name: str):
        if isinstance(name, str):
            self.input_ids_field = name

    def prepare_input(self, *args, **kwargs):
        pass

    def is_img(self, obj):
        if isinstance(obj, Image.Image):
            return obj
        elif isinstance(obj, str):
            if is_url(obj):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                        " Safari/537.36"
                    )
                }
                response = requests.get(obj, stream=True, headers=headers)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                try:
                    return Image.open(obj)
                except:
                    return None

    def get_input_token_num(self, input_tokens: str) -> int:
        return len(self.tokenizer(input_tokens, add_special_tokens=False)["input_ids"])


class IdeficsPromptProcessor(LMMPromptProcessor):
    def __init__(self, model_name_or_path):
        self.processor = IdeficsProcessor.from_pretrained(model_name_or_path)
        super().__init__(self.processor.tokenizer, self.processor.image_processor)

    def prepare_input(
        self,
        batch_prompts,
        padding="longest",
        truncation=None,
        max_length=None,
        transform=None,
        add_eos_token=False,
        add_end_of_utterance_token=None,
        debug=False,
        return_tensors="pt",
    ):
        return self.processor(
            batch_prompts,
            padding,
            truncation,
            max_length,
            transform,
            add_eos_token,
            add_end_of_utterance_token,
            debug,
            return_tensors,
        )


class IdeficsInterface(LMMInterface):
    def __init__(
        self,
        model_name_or_path,
        precision,
        model_device,
        prompt_manager,
        instruction,
        image_field,
        label_field,
    ):
        super().__init__(
            precision=precision,
            input_ids_field_name="input_ids",
            prompt_manager=prompt_manager,
            instruction=instruction,
            label_field=label_field,
            image_field=image_field,
        )
        self.processor = IdeficsPromptProcessor(model_name_or_path)
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_name_or_path,
            torch_dtype=self.data_type,
        ).to(model_device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.image_processor = self.processor.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id

        self.fake_token = "<fake_token_around_image>"
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)