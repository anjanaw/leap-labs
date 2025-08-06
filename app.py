import os
import gradio as gr

from PIL import Image

from lib.pgd import PGD

def run(image: Image.Image, target_class: int, norm_type: str):
    assert target_class >=0 and target_class <1000, "Target ImageNet class should be between 0 and 999. See https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/ for more details."

    pgd = PGD(use_l2= (norm_type=='L2'))
    # Generate adversarial example using projected gradient descent
    adv_image = pgd.generate_from_pil(image, target_class)
    if adv_image:
        return adv_image, f"Successfully generated targeted adversarial example."
    else:
        return None, f"Failed to generate targeted adversarial example."

def app():
    # run gradio app on browser
    interface = gr.Interface(fn=run,
                             inputs=[
                                    gr.Image(type="pil", label="Source Image"),
                                    gr.Number(label="Target ImageNet class (0-999)."),
                                    gr.Radio(choices=["L-inf", "L2"], label="Normalisation", value="L-inf")
                                ],
                            outputs=[gr.Image(label="Adversarial Image"), gr.Textbox()],
                            flagging_dir="./gradio_out")
    interface.launch()

if __name__ == '__main__':
    app()