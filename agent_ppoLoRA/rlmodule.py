import deepspeed
import hydra
import copy
import pytorch_lightning as pl
import sys
import string
import torch
from torch.distributions import Normal
from deepspeed.ops.adam import DeepSpeedCPUAdam
from loguru import logger
from torch import optim
from transformers import get_cosine_schedule_with_warmup

from lmm_icl_interface import LMMInterface
import wandb
import re
import math
from torch.optim.lr_scheduler import LambdaLR
from inference import generate_answers_fixed_alpha

sys.path.append("/home/s223540177/LIVE-Learnable-In-Context-Vector")
from icv_src.icv_encoder.global_icv_encoder import GlobalICVEncoder
from icv_src.icv_model.icv_intervention import LearnableICVInterventionLMM

class VQAICVModule(pl.LightningModule):
    def __init__(
        self,
        interface: LMMInterface,
        cfg,
        module_cfg,
        lmm_cfg,
        policy_net,
        baseline_net,
        processor,
        gamma=0.99,
        lr=5e-6,
        clip_epsilon=0.2,
        entropy_coef=0.001,
        clip_vloss=True,
        temperature_init_value=1.0,      
        learnable_temperature=True,      
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["interface", "policy_net", "baseline_net"])
        self.cfg = cfg
        self.module_cfg = module_cfg
        self.lmm_cfg = lmm_cfg
        self.interface = interface
        self.policy_net = policy_net
        self.baseline_net = baseline_net
        self.processor = processor
        self.gamma = gamma
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.clip_vloss = clip_vloss

        # Freeze the interface model parameters.
        self.interface.requires_grad_(False)
        if hasattr(self.interface.model, "gradient_checkpointing_enable"):
            self.interface.model.gradient_checkpointing_enable()

        self.icv_model = LearnableICVInterventionLMM(
            interface,
            enable_intervention=True,
            intervention_layer=self.lmm_cfg.intervention_layer,
            layer_format=self.lmm_cfg.layer_format,
            total_layers=self.lmm_cfg.total_layers,
        )

        # Adapt temperature initialization similar to GlobalICVEncoder's alpha.
        self.temperature = torch.nn.Parameter(
            torch.full((1,), temperature_init_value),
            requires_grad=learnable_temperature,
        )

        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "long answer": "",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            ":",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
            ".",
        ]

        # Create a frozen copy of the policy network for storing "old" log probabilities.
        self.policy_net_old = copy.deepcopy(self.policy_net)

        # Freeze its parameters so that it does not receive gradients.
        for param in self.policy_net_old.parameters():
            param.requires_grad = False


    def extract_features(self, query_x_input):
        if type(query_x_input) == dict:
            last_hidden_state = self.interface.model(
                **query_x_input, return_dict=True, output_hidden_states=True
            ).hidden_states[-1]
            weights_for_non_padding = query_x_input["attention_mask"].to(last_hidden_state.device) * \
                torch.arange(start=1, end=last_hidden_state.shape[1] + 1, device=last_hidden_state.device).unsqueeze(0)
            sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
            num_of_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1, keepdim=True)
            sentence_embeddings = sum_embeddings / num_of_non_padding_tokens
            return sentence_embeddings.to(dtype=torch.float32)
        else:
            last_hidden_state = self.interface.model(
                **query_x_input, return_dict=True, output_hidden_states=True
            ).hidden_states[-1]
            weights_for_non_padding = query_x_input.attention_mask.to(last_hidden_state.device) * \
                torch.arange(start=1, end=last_hidden_state.shape[1] + 1, device=last_hidden_state.device).unsqueeze(0)
            sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
            num_of_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1, keepdim=True)
            sentence_embeddings = sum_embeddings / num_of_non_padding_tokens
            return sentence_embeddings.to(dtype=torch.float32)

    def compute_reward(self, query_inputs, inputs, icl_logits, icl_context_mask, weight_update, answer_text, query_x_text, query_x_length, instruction, interface):
        self.icv_model.toggle_intervention(True)

        prompts = [[instruction]] if instruction else [[]]
        prompts[0].extend(query_x_text)
        inputs_processed = self.processor.prepare_input(prompts)
        inputs_processed = {k: v.to(self.device) for k, v in inputs_processed.items()}
        
        B, total_layers, hidden_size = weight_update.shape
        update_weight = weight_update  # Already (B, total_layers, hidden_size)
        generated, rl_logit, rl_mask = generate_answers_fixed_alpha(
            inputs=inputs_processed,
            query_inputs=query_inputs,
            model=self.icv_model,
            processor=self.processor,
            total_layers=total_layers,
            hidden_size=hidden_size,
            query_x_length=query_x_length,
            in_context_vector=update_weight,
            interface=interface,
        )

        generated_text = generated[0] if isinstance(generated, list) else generated
        generated_text = generated_text.strip()
        answer_text = answer_text.strip().replace(".", "") if isinstance(answer_text, str) else str(answer_text).strip().replace(".", "")
        answer_text = self.processSpecial(answer_text)
        answer_text = self.processDigitArticle(answer_text)
        answer_text = self.processPunctuation(answer_text)
        answer_text = self.processJargon(answer_text)
        answer_text = answer_text.strip()

        generated_text = self.processSpecial(generated_text)
        generated_text = self.processDigitArticle(generated_text)
        generated_text = self.processPunctuation(generated_text)
        generated_text = self.processJargon(generated_text)
        generated_text = generated_text.strip()

        rewards = torch.tensor(1.0, device="cuda") if generated_text == answer_text else torch.tensor(0.0, device="cuda")
        

        kl_loss = self.calculate_kl_divergence(
            rl_logit[rl_mask].view(-1, rl_logit.shape[-1]),
            icl_logits[icl_context_mask].view(-1, icl_logits.shape[-1]),
        )

        # final_reward = rewards - kl_loss
    
        return -kl_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """
        query_inputs: question + answer (no icl)
        inputs: in-context examples + query (with answer)
        query_x_input: only question
        query_x_text: only question (text)
        """
        query_inputs = batch["query_inputs"]
        inputs = batch["inputs"]
        query_x_input = batch["query_x_input"]
        query_x_text = batch["query_x_text"][0]
        answer_text = batch["answer_text"]
        in_context_length = batch["in_context_length"]
        query_x_length = batch["query_x_length"]

        state = self.extract_features(query_x_input)
        loss = 0.0  # Accumulate loss over n_actions
        n_actions = self.cfg.n_actions

        # Get weight update and related outputs from policy_net
        weight_update, _, entropy, raw_edit = self.policy_net(state)

        # Evaluate old and new log probabilities for the actions
        with torch.no_grad():
            old_log_prob, _ = self.policy_net_old.evaluate_actions(state, raw_edit)
        new_log_prob, _ = self.policy_net.evaluate_actions(state, raw_edit)

        # Toggle intervention and obtain logits from icv_model
        self.icv_model.toggle_intervention(False)
        icl_logits = self.icv_model(**inputs)["logits"] # this is the distribution we aim to match
        icl_context_mask = self.get_mask(inputs, in_context_length)

        # Compute the probability ratio (ensure clip_epsilon is defined)
        log_prob_diff = new_log_prob - old_log_prob
        ratio = torch.exp(log_prob_diff)
        ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # Compute policy clip fraction for logging
        policy_clip_frac = ((ratio < (1 - self.clip_epsilon)) | (ratio > (1 + self.clip_epsilon))).float().mean()

        # Compute the reward for the current action
        reward, kl_mean_loss = self.compute_reward(
            query_inputs, inputs, icl_logits, icl_context_mask, weight_update,
            answer_text, query_x_text, query_x_length, self.cfg.prompt.instruction, self.interface
        )

        # Compute baseline and advantage for the current state-action pair
        baseline = self.baseline_net(state)
        advantage = reward - baseline

        # Compute the policy loss (PG loss) for this action
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        pg_loss = -torch.min(surr1, surr2).mean()

        # Compute the value loss
        baseline_new = self.baseline_net(state)
        v_loss_unclipped = (baseline_new - reward) ** 2
        clipped_diff = torch.clamp(baseline_new - baseline, -self.clip_epsilon, self.clip_epsilon)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (baseline + clipped_diff - reward) ** 2).mean()

        # Compute entropy loss
        entropy_loss = -self.entropy_coef * entropy

        # Total loss for this action
        total_action_loss = pg_loss + v_loss + entropy_loss + kl_mean_loss
        loss += total_action_loss

        # Log metrics for this action if needed
        wandb.log({
            "batch_idx": batch_idx,
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
            "loss/policy": pg_loss,
            "loss/value": v_loss,
            "loss/entropy": entropy_loss,
            "loss/total": loss,
            "loss/kl": kl_mean_loss,
            "reward": reward.mean(),
            "advantage": advantage.mean(),
            "ratio": ratio.mean(),
            "policy/clipfrac_avg": policy_clip_frac,
        })

        # print all the values to find nan
        print(f"batch_idx: {batch_idx}, learning_rate: {self.trainer.optimizers[0].param_groups[0]['lr']}, loss/policy: {pg_loss}, loss/value: {v_loss}, loss/entropy: {entropy_loss}, loss/kl: {kl_mean_loss}, loss/total: {loss}, reward: {reward}, advantage: {advantage}, ratio: {ratio}")

        # Accumulate the loss from the current action
        # print(f"batch_idx: {batch_idx}, learning_rate: {self.trainer.optimizers[0].param_groups[0]['lr']}, loss/policy: {pg_loss}, loss/value: {v_loss}, loss/entropy: {entropy_loss}, loss/kl: {kl_mean_loss}, loss/total: {loss}, reward: {reward}, advantage: {advantage}, ratio: {ratio}") 
        
        # Return the accumulated loss so that Lightning can call backward() and step() automatically.
        return loss

    def on_train_epoch_end(self):
        # Update the old policy network at the end of each epoch so that it tracks the current policy.
        self.policy_net_old.load_state_dict(self.policy_net.state_dict())



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.baseline_net.parameters()), 
            lr=self.lr
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def calculate_kl_divergence(self, stu_logits, tea_logits):
        stu_logits /= self.temperature
        tea_logits /= self.temperature
        return (
            (
                tea_logits.softmax(dim=1)
                * (
                    (tea_logits.softmax(dim=1) + self.module_cfg.kl_eps).log()
                    - (stu_logits.softmax(dim=1) + self.module_cfg.kl_eps).log()
                )
            )
            .sum(dim=1)
            .mean()
        ) * self.temperature**2

    def get_mask(self, inputs, mask_length):
        mask_shape = inputs[self.interface.input_ids_field_name].shape
        bs, seq_len = mask_shape
        device = inputs[self.interface.input_ids_field_name].device
        sequence_indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
        )
        mask = sequence_indices >= mask_length.unsqueeze(dim=1)
        mask[
            inputs[self.interface.input_ids_field_name]
            == self.interface.tokenizer.pad_token_id
        ] = False
        return mask

    def processPunctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def processSpecial(self, inText):
        return inText.replace("\n", "").replace("\t", "")
    
    def processJargon(self, inText):
        return inText.replace("question", "").replace("long answer", "")
    
    



# -----------------------------
# Helper: Construct Weight Update using Outer Product and Normalization
# -----------------------------
def construct_weight_update(layer_coeffs, edit_coeffs, target_std=0.1, eps=1e-6):
    """
    Given:
      - layer_coeffs: (B, num_actions, total_layers, 1)
      - edit_coeffs: (B, num_actions, 1, hidden_size)
      
    Computes the outer product to yield a weight update of shape 
              (B, num_actions, total_layers, hidden_size),
    then normalizes each row vector (for each layer per action) to have mean = 0 and std = target_std.
    """
    # Outer product.
    weight_update = torch.matmul(layer_coeffs, edit_coeffs)  # (B, num_actions, total_layers, hidden_size)
    return weight_update

def cosine_annealing_with_warmup(epoch, warmup_epochs=10, total_epochs=800):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Linear warmup
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return cosine_decay
