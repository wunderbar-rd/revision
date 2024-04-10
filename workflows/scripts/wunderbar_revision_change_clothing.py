import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "samples1",
    help='Argument 0, input `samples` for node "Upscale Latent By" id 215 (autogenerated)',
)

parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
        )
        ordered_args = dict(zip(["samples1"], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        jwinteger = NODE_CLASS_MAPPINGS["JWInteger"]()
        jwinteger_40 = jwinteger.execute()

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_44 = loadimage.load_image(
            image="talent_miki_fujimoto_0421_wunderbar_0870_skin_mask.jpg"
        )

        loadimage_48 = loadimage.load_image(
            image="talent_miki_fujimoto_0421_wunderbar_0870_person_mask.jpg"
        )

        loadimage_52 = loadimage.load_image(
            image="talent_miki_fujimoto_0421_wunderbar_0870_top_mask.jpg"
        )

        loadimage_56 = loadimage.load_image(
            image="talent_miki_fujimoto_0421_wunderbar_0870_bottom_mask.jpg"
        )

        loadimage_60 = loadimage.load_image(
            image="talent_miki_fujimoto_0421_wunderbar_0870_clothes_mask.jpg"
        )

        loadimage_64 = loadimage.load_image(
            image="talent_miki_fujimoto_0421_wunderbar_0870.jpg"
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_138 = controlnetloader.load_controlnet(
            control_net_name="control_v11f1p_sd15_depth_fp16.safetensors"
        )

        controlnetloader_142 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_openpose_fp16.safetensors"
        )

        controlnetloader_146 = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_canny_fp16.safetensors"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_153 = vaeloader.load_vae(
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
        )

        constrainimagepysssss = NODE_CLASS_MAPPINGS["ConstrainImage|pysssss"]()
        constrainimagepysssss_49 = constrainimagepysssss.constrain_image(
            max_width=get_value_at_index(jwinteger_40, 0),
            max_height=get_value_at_index(jwinteger_40, 0),
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(loadimage_52, 0),
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_50 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(constrainimagepysssss_49, 0)
        )

        invertmask_segment_anything = NODE_CLASS_MAPPINGS[
            "InvertMask (segment anything)"
        ]()
        invertmask_segment_anything_66 = invertmask_segment_anything.main(
            mask=get_value_at_index(imagetomask_50, 0)
        )

        mask_to_image_mtb = NODE_CLASS_MAPPINGS["Mask To Image (mtb)"]()
        mask_to_image_mtb_65 = mask_to_image_mtb.render_mask(
            color="#05037c",
            background="#ffffff",
            mask=get_value_at_index(invertmask_segment_anything_66, 0),
        )

        grayscale_image_wlsh = NODE_CLASS_MAPPINGS["Grayscale Image (WLSH)"]()
        grayscale_image_wlsh_78 = grayscale_image_wlsh.make_grayscale(
            original=get_value_at_index(loadimage_64, 0)
        )

        constrainimagepysssss_79 = constrainimagepysssss.constrain_image(
            max_width=get_value_at_index(jwinteger_40, 0),
            max_height=get_value_at_index(jwinteger_40, 0),
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(grayscale_image_wlsh_78, 0),
        )

        image_blending_mode = NODE_CLASS_MAPPINGS["Image Blending Mode"]()
        image_blending_mode_81 = image_blending_mode.image_blending_mode(
            mode="multiply",
            blend_percentage=0.9500000000000001,
            image_a=get_value_at_index(mask_to_image_mtb_65, 0),
            image_b=get_value_at_index(constrainimagepysssss_79, 0),
        )

        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        imagecompositemasked_84 = imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(image_blending_mode_81, 0),
            source=get_value_at_index(constrainimagepysssss_49, 0),
            mask=get_value_at_index(imagetomask_50, 0),
        )

        constrainimagepysssss_61 = constrainimagepysssss.constrain_image(
            max_width=get_value_at_index(jwinteger_40, 0),
            max_height=get_value_at_index(jwinteger_40, 0),
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(loadimage_64, 0),
        )

        image_overlay = NODE_CLASS_MAPPINGS["Image Overlay"]()
        image_overlay_109 = image_overlay.apply_overlay_image(
            overlay_resize="None",
            resize_method="area",
            rescale_factor=1,
            width=512,
            height=512,
            x_offset=0,
            y_offset=0,
            rotation=0,
            opacity=0,
            base_image=get_value_at_index(imagecompositemasked_84, 0),
            overlay_image=get_value_at_index(constrainimagepysssss_61, 0),
            optional_mask=get_value_at_index(invertmask_segment_anything_66, 0),
        )

        vaeencodetiled = NODE_CLASS_MAPPINGS["VAEEncodeTiled"]()
        vaeencodetiled_156 = vaeencodetiled.encode(
            tile_size=512,
            pixels=get_value_at_index(image_overlay_109, 0),
            vae=get_value_at_index(vaeloader_153, 0),
        )

        constrainimagepysssss_53 = constrainimagepysssss.constrain_image(
            max_width=get_value_at_index(jwinteger_40, 0),
            max_height=get_value_at_index(jwinteger_40, 0),
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(loadimage_56, 0),
        )

        imagetomask_54 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(constrainimagepysssss_53, 0)
        )

        invertmask_segment_anything_67 = invertmask_segment_anything.main(
            mask=get_value_at_index(imagetomask_54, 0)
        )

        mask_to_image_mtb_87 = mask_to_image_mtb.render_mask(
            color="#042234",
            background="#ffffff",
            mask=get_value_at_index(invertmask_segment_anything_67, 0),
        )

        image_blending_mode_88 = image_blending_mode.image_blending_mode(
            mode="multiply",
            blend_percentage=0.9500000000000001,
            image_a=get_value_at_index(mask_to_image_mtb_87, 0),
            image_b=get_value_at_index(constrainimagepysssss_79, 0),
        )

        imagecompositemasked_89 = imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(image_blending_mode_88, 0),
            source=get_value_at_index(constrainimagepysssss_53, 0),
            mask=get_value_at_index(imagetomask_54, 0),
        )

        image_overlay_122 = image_overlay.apply_overlay_image(
            overlay_resize="None",
            resize_method="area",
            rescale_factor=1,
            width=512,
            height=512,
            x_offset=0,
            y_offset=0,
            rotation=0,
            opacity=0,
            base_image=get_value_at_index(imagecompositemasked_84, 0),
            overlay_image=get_value_at_index(imagecompositemasked_89, 0),
            optional_mask=get_value_at_index(invertmask_segment_anything_66, 0),
        )

        maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        maskcomposite_126 = maskcomposite.combine(
            x=0,
            y=0,
            operation="add",
            destination=get_value_at_index(invertmask_segment_anything_67, 0),
            source=get_value_at_index(invertmask_segment_anything_66, 0),
        )

        image_overlay_129 = image_overlay.apply_overlay_image(
            overlay_resize="None",
            resize_method="area",
            rescale_factor=1,
            width=512,
            height=512,
            x_offset=0,
            y_offset=0,
            rotation=0,
            opacity=0,
            base_image=get_value_at_index(image_overlay_122, 0),
            overlay_image=get_value_at_index(constrainimagepysssss_61, 0),
            optional_mask=get_value_at_index(maskcomposite_126, 0),
        )

        vaeencodetiled_166 = vaeencodetiled.encode(
            tile_size=512,
            pixels=get_value_at_index(image_overlay_129, 0),
            vae=get_value_at_index(vaeloader_153, 0),
        )

        constrainimagepysssss_57 = constrainimagepysssss.constrain_image(
            max_width=get_value_at_index(jwinteger_40, 0),
            max_height=get_value_at_index(jwinteger_40, 0),
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(loadimage_60, 0),
        )

        imagetomask_58 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(constrainimagepysssss_57, 0)
        )

        invertmask_segment_anything_68 = invertmask_segment_anything.main(
            mask=get_value_at_index(imagetomask_58, 0)
        )

        mask_to_image_mtb_91 = mask_to_image_mtb.render_mask(
            color="#003e8f",
            background="#ffffff",
            mask=get_value_at_index(invertmask_segment_anything_68, 0),
        )

        image_blending_mode_92 = image_blending_mode.image_blending_mode(
            mode="multiply",
            blend_percentage=0.9500000000000001,
            image_a=get_value_at_index(mask_to_image_mtb_91, 0),
            image_b=get_value_at_index(constrainimagepysssss_79, 0),
        )

        imagecompositemasked_93 = imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(image_blending_mode_92, 0),
            source=get_value_at_index(constrainimagepysssss_57, 0),
            mask=get_value_at_index(imagetomask_58, 0),
        )

        image_overlay_100 = image_overlay.apply_overlay_image(
            overlay_resize="None",
            resize_method="area",
            rescale_factor=1,
            width=512,
            height=512,
            x_offset=0,
            y_offset=0,
            rotation=0,
            opacity=0,
            base_image=get_value_at_index(imagecompositemasked_93, 0),
            overlay_image=get_value_at_index(constrainimagepysssss_61, 0),
            optional_mask=get_value_at_index(invertmask_segment_anything_68, 0),
        )

        vaeencodetiled_173 = vaeencodetiled.encode(
            tile_size=512,
            pixels=get_value_at_index(image_overlay_100, 0),
            vae=get_value_at_index(vaeloader_153, 0),
        )

        image_overlay_112 = image_overlay.apply_overlay_image(
            overlay_resize="None",
            resize_method="area",
            rescale_factor=1,
            width=512,
            height=512,
            x_offset=0,
            y_offset=0,
            rotation=0,
            opacity=0,
            base_image=get_value_at_index(imagecompositemasked_89, 0),
            overlay_image=get_value_at_index(constrainimagepysssss_61, 0),
            optional_mask=get_value_at_index(invertmask_segment_anything_67, 0),
        )

        vaeencodetiled_180 = vaeencodetiled.encode(
            tile_size=512,
            pixels=get_value_at_index(image_overlay_112, 0),
            vae=get_value_at_index(vaeloader_153, 0),
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_188 = checkpointloadersimple.load_checkpoint(
            ckpt_name="Realistic_Vision_V6.0_NV_B1.safetensors"
        )

        clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
        clipsetlastlayer_189 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-1,
            clip=get_value_at_index(checkpointloadersimple_188, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_190 = cliptextencode.encode(
            text="woman wearing navy blue elegant fashion blouse and gray floral skirt ",
            clip=get_value_at_index(clipsetlastlayer_189, 0),
        )

        cliptextencode_191 = cliptextencode.encode(
            text="nude, nsfw", clip=get_value_at_index(clipsetlastlayer_189, 0)
        )

        seed_rgthree = NODE_CLASS_MAPPINGS["Seed (rgthree)"]()
        seed_rgthree_213 = seed_rgthree.main(
            seed=random.randint(1, 2**64), unique_id=16315379179799082811
        )

        latentupscaleby = NODE_CLASS_MAPPINGS["LatentUpscaleBy"]()
        latentupscaleby_215 = latentupscaleby.upscale(
            upscale_method="nearest-exact", scale_by=4, samples=parse_arg(args.samples1)
        )

        maskpreview = NODE_CLASS_MAPPINGS["MaskPreview+"]()
        cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()
        controlnetapply = NODE_CLASS_MAPPINGS["ControlNetApply"]()
        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        setlatentnoisemask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
        impactswitch = NODE_CLASS_MAPPINGS["ImpactSwitch"]()
        vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()
        for q in range(args.queue_size):
            constrainimagepysssss_41 = constrainimagepysssss.constrain_image(
                max_width=get_value_at_index(jwinteger_40, 0),
                max_height=get_value_at_index(jwinteger_40, 0),
                min_width=0,
                min_height=0,
                crop_if_required="no",
                images=get_value_at_index(loadimage_44, 0),
            )

            imagetomask_42 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(constrainimagepysssss_41, 0)
            )

            invertmask_segment_anything_95 = invertmask_segment_anything.main(
                mask=get_value_at_index(imagetomask_42, 0)
            )

            maskpreview_43 = maskpreview.execute(
                mask=get_value_at_index(invertmask_segment_anything_95, 0)
            )

            constrainimagepysssss_45 = constrainimagepysssss.constrain_image(
                max_width=get_value_at_index(jwinteger_40, 0),
                max_height=get_value_at_index(jwinteger_40, 0),
                min_width=0,
                min_height=0,
                crop_if_required="no",
                images=get_value_at_index(loadimage_48, 0),
            )

            imagetomask_46 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(constrainimagepysssss_45, 0)
            )

            invertmask_segment_anything_96 = invertmask_segment_anything.main(
                mask=get_value_at_index(imagetomask_46, 0)
            )

            maskpreview_47 = maskpreview.execute(
                mask=get_value_at_index(invertmask_segment_anything_96, 0)
            )

            maskpreview_51 = maskpreview.execute(
                mask=get_value_at_index(invertmask_segment_anything_66, 0)
            )

            maskpreview_55 = maskpreview.execute(
                mask=get_value_at_index(invertmask_segment_anything_67, 0)
            )

            maskpreview_59 = maskpreview.execute(
                mask=get_value_at_index(invertmask_segment_anything_68, 0)
            )

            cannyedgepreprocessor_149 = cannyedgepreprocessor.execute(
                low_threshold=100,
                high_threshold=200,
                resolution=512,
                image=get_value_at_index(constrainimagepysssss_79, 0),
            )

            controlnetapply_145 = controlnetapply.apply_controlnet(
                strength=0.75,
                conditioning=get_value_at_index(cliptextencode_190, 0),
                control_net=get_value_at_index(controlnetloader_146, 0),
                image=get_value_at_index(cannyedgepreprocessor_149, 0),
            )

            depthanythingpreprocessor_139 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitl14.pth",
                resolution=512,
                image=get_value_at_index(constrainimagepysssss_79, 0),
            )

            controlnetapply_133 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(controlnetapply_145, 0),
                control_net=get_value_at_index(controlnetloader_138, 0),
                image=get_value_at_index(depthanythingpreprocessor_139, 0),
            )

            dwpreprocessor_151 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(constrainimagepysssss_79, 0),
            )

            controlnetapply_141 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(controlnetapply_133, 0),
                control_net=get_value_at_index(controlnetloader_142, 0),
                image=get_value_at_index(dwpreprocessor_151, 0),
            )

            setlatentnoisemask_175 = setlatentnoisemask.set_mask(
                samples=get_value_at_index(vaeencodetiled_173, 0),
                mask=get_value_at_index(invertmask_segment_anything_68, 0),
            )

            setlatentnoisemask_168 = setlatentnoisemask.set_mask(
                samples=get_value_at_index(vaeencodetiled_166, 0),
                mask=get_value_at_index(maskcomposite_126, 0),
            )

            setlatentnoisemask_160 = setlatentnoisemask.set_mask(
                samples=get_value_at_index(vaeencodetiled_156, 0),
                mask=get_value_at_index(invertmask_segment_anything_66, 0),
            )

            setlatentnoisemask_182 = setlatentnoisemask.set_mask(
                samples=get_value_at_index(vaeencodetiled_180, 0),
                mask=get_value_at_index(invertmask_segment_anything_67, 0),
            )

            impactswitch_155 = impactswitch.doit()

            vaedecodetiled_163 = vaedecodetiled.decode(
                tile_size=512,
                samples=get_value_at_index(setlatentnoisemask_160, 0),
                vae=get_value_at_index(vaeloader_153, 0),
            )

            vaedecodetiled_171 = vaedecodetiled.decode(
                tile_size=512,
                samples=get_value_at_index(setlatentnoisemask_168, 0),
                vae=get_value_at_index(vaeloader_153, 0),
            )

            vaedecodetiled_178 = vaedecodetiled.decode(
                tile_size=512,
                samples=get_value_at_index(setlatentnoisemask_175, 0),
                vae=get_value_at_index(vaeloader_153, 0),
            )

            vaedecodetiled_185 = vaedecodetiled.decode(
                tile_size=512,
                samples=get_value_at_index(setlatentnoisemask_182, 0),
                vae=get_value_at_index(vaeloader_153, 0),
            )

            freeu_v2_193 = freeu_v2.patch(
                b1=1.3,
                b2=1.4,
                s1=0.9,
                s2=0.2,
                model=get_value_at_index(checkpointloadersimple_188, 0),
            )

            differentialdiffusion_194 = differentialdiffusion.apply(
                model=get_value_at_index(freeu_v2_193, 0)
            )

            ksampler_192 = ksampler.sample(
                seed=get_value_at_index(seed_rgthree_213, 0),
                steps=40,
                cfg=6,
                sampler_name="dpmpp_2m_sde_gpu",
                scheduler="karras",
                denoise=0.8,
                model=get_value_at_index(differentialdiffusion_194, 0),
                positive=get_value_at_index(controlnetapply_141, 0),
                negative=get_value_at_index(cliptextencode_191, 0),
                latent_image=get_value_at_index(impactswitch_155, 0),
            )

            vaedecodetiled_204 = vaedecodetiled.decode(
                tile_size=512,
                samples=get_value_at_index(ksampler_192, 0),
                vae=get_value_at_index(vaeloader_153, 0),
            )

            ksampler_206 = ksampler.sample(
                seed=get_value_at_index(seed_rgthree_213, 0),
                steps=10,
                cfg=8,
                sampler_name="dpmpp_2m_sde_gpu",
                scheduler="karras",
                denoise=0.3,
                model=get_value_at_index(differentialdiffusion_194, 0),
                positive=get_value_at_index(controlnetapply_141, 0),
                negative=get_value_at_index(cliptextencode_191, 0),
                latent_image=get_value_at_index(ksampler_192, 0),
            )

            vaedecodetiled_209 = vaedecodetiled.decode(
                tile_size=512,
                samples=get_value_at_index(ksampler_206, 0),
                vae=get_value_at_index(vaeloader_153, 0),
            )

            image_comparer_rgthree_211 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(vaedecodetiled_209, 0),
                image_b=get_value_at_index(constrainimagepysssss_61, 0),
            )


if __name__ == "__main__":
    main()
