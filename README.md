---
title: BiRefNet Background Removal Tool
emoji: ✂️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.1 # Adjust this version if necessary
app_file: app.py
license: apache-2.0
models:
  - ZhengPeng7/BiRefNet
---

# BiRefNet: High-Quality Background Removal Tool ✂️

This is a high-performance web application built with **Gradio** that uses the **BiRefNet** model for high-resolution, precise image segmentation and background removal. The application allows users to quickly convert any image (uploaded or via URL) into a transparent PNG cutout.

The core technology is based on the [**BiRefNet**](https://huggingface.co/ZhengPeng7/BiRefNet) architecture, which specializes in **Dichotomous Image Segmentation (DIS)**—the task of accurately separating the foreground object from the background.



---

## ✨ Features

The application is structured into three easy-to-use tabs:

* **🖼️ Image Upload:** Upload a local image file and view the original and the transparent result side-by-side using an `ImageSlider` component.
* **🔗 URL Input:** Paste a public image URL and process the image remotely, displaying the results in a side-by-side `ImageSlider`.
* **💾 File Output:** Upload an image file and receive the final processed image directly as a downloadable PNG file with a transparent background.

---

## ⚙️ Model and Technology

* **Model:** [**ZhengPeng7/BiRefNet**](https://huggingface.co/ZhengPeng7/BiRefNet) (Bilateral Reference for High-Resolution Dichotomous Image Segmentation).
* **Framework:** Built using the `transformers` library for loading the segmentation model and **Gradio** for the interactive user interface.
* **Performance:** The core segmentation logic (`process` function) is accelerated using the `@spaces.GPU` decorator and runs on a CUDA-enabled device (`cuda`) when available, ensuring fast inference times for high-resolution images.
* **Processing Details:**
    * Input images are resized to `1024x1024` and normalized before feeding into the BiRefNet model.
    * The model predicts a **segmentation mask**.
    * The mask is resized back to the original image dimensions and applied as an **alpha channel** to create the final transparent PNG output.

---

## 🚀 How to Run

### On Hugging Face Spaces

1.  **Duplicate the Space:** Click the **"Duplicate Space"** button in the top right corner.
2.  **Adjust Hardware:** If you choose a free CPU tier, performance will be significantly slower. It is recommended to select a **GPU** tier (e.g., A10G Small) for the best experience.
3.  The application will automatically build and launch based on the `app.py` file and the required dependencies in `requirements.txt`.

### Locally

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repo-Link]
    cd [Your-Repo-Name]
    ```
2.  **Install dependencies:** (Make sure you have all required packages like `gradio`, `transformers`, `torch`, `Pillow`, `loadimg`, `spaces`, and `torchvision` installed).
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will launch in your browser at `http://localhost:7860`.

---

## 📝 Dependencies

The main dependencies should be listed in a `requirements.txt` file in your repository:

```text
gradio
transformers
torch
torchvision
Pillow
spaces
loadimg # Assuming 'loadimg' is a local or internal helper utility
