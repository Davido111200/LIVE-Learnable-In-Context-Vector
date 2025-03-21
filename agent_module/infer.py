import datetime
import json
import os
import random
from pathlib import Path
import pytorch_lightning as pl

import hydra
import more_itertools
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
import sys
sys.path.append('/home/s223540177/LIVE-Learnable-In-Context-Vector')

from icv_src.icv_model.icv_intervention import LearnableICVInterventionLMM
from icv_src.metrics import compute_cider, compute_vqa_accuracy
from lmm_icl_interface import LMMPromptManager
from rlmodule import VQAICVModule  
from utils import get_icv_cpk_path, get_inference_paths, init_dataset, init_interface
from cleanmain import OuterProductPolicyNet


def evaluate_caption(results_dict, model_name, val_ann_path, post_process_fun):
    pred_coco = []
    for idx in results_dict:
        pred_coco.append(
            {
                "image_id": results_dict[idx]["image_id"],
                "caption": post_process_fun(
                    results_dict[idx]["prediction"], model_name
                ),
            }
        )
    cider_score = compute_cider(pred_coco, val_ann_path)
    return cider_score * 100


def evaluate_vqa(
    results_dict,
    model_name,
    val_ques_path,
    val_ann_path,
    post_process_fun,
):
    preds = []
    for idx in results_dict:
        preds.append(
            {
                "answer": post_process_fun(results_dict[idx]["prediction"], model_name)
                .replace("\n", "")
                .strip(),
                "question_id": results_dict[idx]["question_id"],
            }
        )
    acc = compute_vqa_accuracy(preds, val_ques_path, val_ann_path)
    return acc


@hydra.main(config_path="../config", config_name="inference.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    #save results
    save_path = f"/home/s223540177/LIVE-Learnable-In-Context-Vector/results/inference/idefics-9b/vqav2/vqav2_idefics_icv/result_rl.json"

    # if os.path.exists(save_path):
    #     print("File already exists")
    #     split = "validation"
    #     ds, post_process_fun = init_dataset(cfg, split)
    #     results_dict = json.load(open(save_path))
    #     if cfg.data_cfg.task.task_name == "vqa":
    #         acc = evaluate_vqa(
    #             results_dict,
    #             cfg.lmm.name,
    #             cfg.data_cfg.task.datasets.val_ques_path,
    #             cfg.data_cfg.task.datasets.val_ann_path,
    #             post_process_fun,
    #         )
    #         logger.info(f"{cfg.run_name} ACC: {acc['overall']}")
    #         results_dict[base_info + "icv result"] = acc
    #     elif cfg.data_cfg.task.task_name == "caption":
    #         cider = evaluate_caption(
    #             results_dict,
    #             cfg.lmm.name,
    #             cfg.data_cfg.task.datasets.val_coco_annotation_file,
    #             post_process_fun,
    #         )
    #         logger.info(f"{cfg.run_name} CIDEr: {cider}")
    #     return

    input_dim = 4096  # Adjust as needed.
    total_layers = cfg.lmm.total_layers
    hidden_size = cfg.lmm.hidden_size

    # NOTE: reeval here
    model_path = '/home/s223540177/LIVE-Learnable-In-Context-Vector/saves/outer_policy_net.pth'
    state_dict = torch.load(model_path)

    policy_net = OuterProductPolicyNet(
        input_dim=input_dim, 
        total_layers=total_layers, 
        hidden_size=hidden_size, 
        num_actions=1
    )
    policy_net.load_state_dict(state_dict)

    policy_net.eval()

    logger.info(f"begin run: {cfg.run_name}")
    result_dir = Path(cfg.result_dir)
    model_name = cfg.lmm.model_name

    save_dir, meta_info_dir, metric_file_path = get_inference_paths(
        result_dir=result_dir,
        model_name=model_name,
        dataset_name=cfg.data_cfg.task.datasets.name,
        run_name=cfg.run_name,
    )

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if not meta_info_dir.exists():
        meta_info_dir.mkdir()

    if not metric_file_path.exists():
        result_dict = {}
    elif cfg.re_eval:
        result_dict = json.load(open(metric_file_path))
        logger.info(f"{metric_file_path} exists. LOADING...")
    elif not cfg.re_eval:
        logger.info(f"{metric_file_path} exists, EXIT...")
        # return

    alpha = None
    prompt_manager, interface, processor = init_interface(cfg)
    
    split = "validation"
    base_info = f"{str(datetime.datetime.now())}-{cfg.test_num=}-"
    if cfg.test_icl:
        split = None

    ds, post_process_fun = init_dataset(cfg, split)

    if cfg.test_icl:
        val_ds = ds["validation"]
        if cfg.train_num != -1:
            train_ds = ds["train"].select(
                random.sample(range(len(ds["train"])), cfg.train_num)
            )
        else:
            train_ds = ds["train"]
    else:
        val_ds = ds

    if cfg.test_num != -1:
        val_ds = val_ds.select(range(cfg.test_num))

    icv_model = LearnableICVInterventionLMM(
        interface,
        enable_intervention=True,
        intervention_layer=cfg.lmm.intervention_layer,
        layer_format=cfg.lmm.layer_format,
        total_layers=cfg.lmm.total_layers,
    )
    icv_model.toggle_intervention(True)
    logger.info(f"{icv_model.intervention_enabled=}")

    module = VQAICVModule(interface, cfg, cfg.icv_module, cfg.lmm, policy_net, None, processor)

    results_dict = rl_inference(
            val_ds=val_ds,
            icv_model=icv_model,
            prompt_manager=prompt_manager,
            processor=processor,
            bs=cfg.bs,
            generate_kwargs=cfg.generate_kwargs,
            instruction=cfg.prompt.instruction,
            policy_net=policy_net,
            module=module,
            alpha=alpha,
        )
    
    with open(save_path, "w") as f:
        json.dump(results_dict, f)

    if cfg.data_cfg.task.task_name == "vqa":
        acc = evaluate_vqa(
            results_dict,
            cfg.lmm.name,
            cfg.data_cfg.task.datasets.val_ques_path,
            cfg.data_cfg.task.datasets.val_ann_path,
            post_process_fun,
        )
        logger.info(f"{cfg.run_name} ACC: {acc['overall']}")
        result_dict[base_info + "icv result"] = acc
    elif cfg.data_cfg.task.task_name == "caption":
        cider = evaluate_caption(
            results_dict,
            cfg.lmm.name,
            cfg.data_cfg.task.datasets.val_coco_annotation_file,
            post_process_fun,
        )
        logger.info(f"{cfg.run_name} CIDEr: {cider}")



@torch.inference_mode()
def rl_inference(
    val_ds,
    icv_model,
    prompt_manager: LMMPromptManager,
    processor,
    bs,
    generate_kwargs,
    instruction="",
    policy_net=None,
    module=None,
    in_context_vector=None,
    alpha=None,
):
    assert policy_net is not None, "Policy net is not provided"
    assert module is not None, "Module is not provided"
    results_dict = {}

    index = 0
    icv_model = icv_model.cuda()

    for batch in more_itertools.chunked(tqdm(val_ds, total=len(val_ds)), 1):

        if instruction:
            prompts = [[instruction] for _ in range(bs)]
        else:
            prompts = [[] for _ in range(bs)]
        for i, sample in enumerate(batch):

            prompts[i].extend(
                [
                    sample["image"],
                    prompt_manager.gen_query_text_without_label(sample),
                ]
            )

        query_inputs = processor.prepare_input(prompts)
        query_inputs = {k: v.to(icv_model.lmm.device) for k, v in query_inputs.items()}

        state = module.extract_features(query_inputs)
        policy_net = policy_net.cuda()
        weight_update, _, _, _ = policy_net(state.cuda())
        weight_update = weight_update.squeeze(1)

        # create a weight update with all elements set to 0
        weight_update = torch.zeros_like(weight_update)
        generated = generate_answers(
            inputs=query_inputs,
            model=icv_model,
            processor=processor,
            generate_kwargs=generate_kwargs,
            in_context_vector=weight_update,
        )
        

        for i in range(len(batch)):
            batch[i].pop("image")
            results_dict[index] = {
                "prediction": generated[i],
                **batch[i],
            }
            index += 1
        print("Generated: ", generated[i])
        
    return results_dict



@torch.inference_mode()
def generate_answers(
    inputs,
    model,
    processor,
    generate_kwargs,
    in_context_vector=None,
    alpha=None,
):
    icv = in_context_vector

    model = model.cuda()
    generated_out = model.generate(**inputs, **generate_kwargs, icv=icv)
    prompt_len = int(inputs["attention_mask"].shape[1])
    outputs = generated_out.tolist()


    generated = processor.tokenizer.batch_decode(
        [output[prompt_len:] for output in outputs],
        skip_special_tokens=True,
    )


    return generated

@torch.inference_mode()
def generate_answers_fixed_alpha(
    inputs,
    model,
    processor,
    generate_kwargs,
    total_layers,
    hidden_size,
    in_context_vector=None,
):
    icv = in_context_vector
    model = model.cuda()
    generated_out = model.generate(**inputs, **generate_kwargs, icv=icv)
    prompt_len = int(inputs["attention_mask"].shape[1])
    outputs = generated_out.tolist()

    generated = processor.tokenizer.batch_decode(
        [output[prompt_len:] for output in outputs],
        skip_special_tokens=True,
    )
    return generated



if __name__ == "__main__":
    load_dotenv()
    torch.set_grad_enabled(False)
    main()
