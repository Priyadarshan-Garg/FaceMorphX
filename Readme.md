# FaceMorphX

## Description

- This project performs real-time face swapping on a webcam feed and allows you to overlay stickers on detected faces, accelerated by GPU processing.
- **Demo** :
    - [FaceFusion](demo/FaceSwap.mp4)
    - [FaceFx](demo/FaceSticker.mp4)

## Features

* **Real-time Face Swapping 📹:** Swaps faces in a live webcam feed.
* **Sticker Overlay 🥰:** Overlays a sticker on the swapped face.
* **GPU Accelerated:** Utilizes NVIDIA CUDA and cuDNN for fast processing.
* **Multiple Face Support:** Detects and swaps faces even when multiple people are in the frame.
* **Custom Source Image:** Allows you to use your own images for face swapping.
* **Dynamic Source Face Selection:** (If implemented) Switches between multiple source faces.
* **User-Friendly Control:** Press `Q` to exit the application. Press `S` to switch source faces (if implemented).

## Technologies Used

* **Python:** (Core language)
* **OpenCV:** (Webcam access, image processing, GUI display)
* **InsightFace:** (Face detection, alignment, landmark prediction)
* **ONNX Runtime:** (Efficient model inference, especially with GPU acceleration)
* **MediaPipe:** (Face detection, replacement - *Note: Please clarify the specific MediaPipe usage. Is it for sticker overlay?*)
* **NumPy:** (Numerical operations, array handling)
* **NVIDIA CUDA Toolkit & cuDNN:** (For GPU acceleration)

## Setup and Installation

1.  **Fork and then Clone the Repository:**

    ```bash
    git clone <your_forked_repo_link>
    cd <FaceMorphX>
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**

    * **Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    * **macOS/Linux:**

        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **NVIDIA GPU Setup (If Available):**

    #### Method 1: Recommended

    * Install the **latest NVIDIA Graphics Drivers.**
    * Install the **NVIDIA CUDA Toolkit** (version compatible with `onnxruntime-gpu`, e.g., 12.x for ONNX Runtime 1.17.1).
    * Install the **cuDNN Library** (version compatible with your CUDA Toolkit, e.g., 9.x for CUDA 12.x). Copy the contents of the cuDNN `bin`, `include`, and `lib` folders into the corresponding CUDA Toolkit directories.
    * Verify that the following directories are added to your system's `PATH` environment variable:
        * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
        * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp`
        * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib`
    * **Restart your system** after CUDA/cuDNN setup.
    * [Download inswapper\_128.onnx](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view) and place it in the appropriate directory (usually `~/.insightface/models/`).

    #### Method 2: Video Tutorial (Optional)

    * [Installation Video](https://www.youtube.com/watch?v=nATRPPZ5dGE) 

6.  **Usage**

    * Place your source image (e.g., `RDJ.jpeg`) in the `src/dynamicFaceBlender/` directory or specify the correct path in the script.
    * Run the script:

        ```bash
        python src/dynamicFaceBlender/FaceFusion.py
        ```

    * **Controls:**
        * Press `Q` to **STOP** the application.
        * Press `S` to **Switch** the source face.

## Troubleshooting

* **`LoadLibrary failed with error 126` or `cudnn64_9.dll is missing`**

    * **Reason:** Incompatible or missing CUDA/cuDNN files, or incorrect `PATH` setup.
    * **Solution:** Ensure you have the correct CUDA Toolkit (e.g., 12.x) and cuDNN (e.g., 9.x) versions installed. Verify that cuDNN files are correctly placed in the CUDA Toolkit directories and that `PATH` variables are set up correctly. A complete reinstallation of the CUDA Toolkit and a fresh cuDNN merge might be necessary.
                 Or else try changing *ctx_id* to **-1** for `CPU Proccesing`

* **`cv2.error: The function is not implemented...`**

    * **Reason:** Issue with the OpenCV GUI backend.
    * **Solution:** Reinstall `opencv-python`:

        ```bash
        pip uninstall opencv-python opencv-contrib-python
        pip install opencv-python
        ```

* **`No face detected in Source Image.`**

    * **Reason:** The source image is not found, or no face is detected in the image.
    * **Solution:** Double-check the path to your source image in the script. Ensure the image contains a clear, well-lit face. Placing the source image in the same directory as the Python script can simplify the path.

* **`numpy.ndarray' object has no attribute 'kps`**

    * **Reason:** InsightFace models may be corrupt or not properly loaded.
    * **Solution:** Delete the `C:\Users\<YourUsername>\.insightface\models\buffalo_l\` folder and re-run the script to force a re-download of the models.

## Contribution

I'm currently working on transferring face coordinates to a MetaHuman character's face rig in Unreal Engine. Feel free to open issues or pull requests!