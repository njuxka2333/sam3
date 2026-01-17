import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model

bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path="/home/houjunlin/.cache/huggingface/hub/models--facebook--sam3/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"

model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
processor = Sam3Processor(model)
# Load an image
image = Image.open("/jhcnas1/houjunlin/fxx/sam3_datasets/pathocell/images/reg001_A.hdf_TILE_2.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="Cell")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(masks.shape, boxes.shape, scores.shape)

import matplotlib.pyplot as plt
import numpy as np

img_np = np.array(image)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_np)

for i in range(len(scores)):
    mask = masks[i].cpu().numpy().squeeze()  # 去掉多余的维度，变成 (224, 224)
    box = boxes[i].cpu().numpy()
    score = scores[i].item()

    # 叠加掩码
    ax.imshow(np.ma.masked_where(mask == 0, mask),
              cmap="jet", alpha=0.4)

    # 绘制边界框
    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)

    # 显示分数
    ax.text(x1, y1 - 5, f"{score:.2f}",
            color="yellow", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5))

ax.axis("off")
plt.tight_layout()
plt.savefig("Example_output.png", dpi=300)
plt.close()


#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]