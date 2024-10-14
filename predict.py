# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from weights_downloader import WeightsDownloader

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]
mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow_api.json"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        weights_downloader = WeightsDownloader()

        weights = [
            "vae/ae.safetensors",
            "clip/clip_l.safetensors",
            "clip/t5xxl_fp8_e4m3fn.safetensors",
            "tensorrt/flux1-dev_DYN_H100-b-1-1-1-h-512-1536-1024-w-512-1536-1024.engine",
            "tensorrt/flux1-schnell_DYN_H100-b-1-1-1-h-512-1536-1024-w-512-1536-1024.engine",
        ]

        for weight in weights:
            weights_downloader.download(
                weight,
                f"https://weights.replicate.delivery/default/comfy-ui/{weight}.tar",
                f"ComfyUI/models/{weight.split('/')[0]}",
            )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        workflow["281"]["inputs"]["width"] = kwargs["width"]
        workflow["281"]["inputs"]["height"] = kwargs["height"]
        workflow["135"]["inputs"]["width"] = kwargs["width"]
        workflow["135"]["inputs"]["height"] = kwargs["height"]

        sampler = workflow["271"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["cfg"]

        engine_loader = workflow["279"]["inputs"]

        if kwargs["model"] == "flux1-dev":
            engine_loader["unet_name"] = (
                "flux1-dev_DYN_H100-b-1-1-1-h-512-1536-1024-w-512-1536-1024.engine"
            )
            engine_loader["model_type"] = "flux_dev"

            sampler["steps"] = 28
        elif kwargs["model"] == "flux1-schnell":
            engine_loader["unet_name"] = (
                "flux1-schnell_DYN_H100-b-1-1-1-h-512-1536-1024-w-512-1536-1024.engine"
            )
            engine_loader["model_type"] = "flux_schnell"
            sampler["steps"] = 4
        else:
            raise ValueError(f"Invalid model: {kwargs['model']}")

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        model: str = Input(
            choices=["flux1-dev", "flux1-schnell"],
            default="flux1-dev",
            description="The model to use for inference.",
        ),
        aspect_ratio: str = Input(
            choices=list(ASPECT_RATIOS.keys()),
            default="1:1",
            description="The aspect ratio of your output image. This value is ignored if you are using an input image.",
        ),
        cfg: float = Input(
            description="The guidance scale tells the model how similar the output should be to the prompt.",
            le=20,
            ge=0,
            default=3.5,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        width, height = ASPECT_RATIOS.get(aspect_ratio)

        self.update_workflow(
            workflow,
            prompt=prompt,
            seed=seed,
            cfg=cfg,
            width=width,
            height=height,
            model=model,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        files = self.comfyUI.get_files(OUTPUT_DIR)
        if len(files) == 0:
            raise Exception("No output")

        return optimise_images.optimise_image_files(
            output_format, output_quality, files
        )
