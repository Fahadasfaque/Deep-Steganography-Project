# 🕵️‍♂️ StegoGenius: Deep Steganography Dashboard

> **Invisible Data Hiding Powered by Stego-GAN & Modern Web Technologies**

![StegoGenius Banner](https://res.cloudinary.com/dhclr2ufn/image/upload/v1770816123/screencapture-127-0-0-1-8000-2026-02-11-18_48_17_jp9lnz.png)

## 🌟 Overview

**StegoGenius** is a state-of-the-art implementation of Deep Steganography, using Generative Adversarial Networks (GANs) to hide secret images inside cover images with near-perfect invisibility.

Unlike traditional LS (Least Significant Bit) methods that are easily detected, StegoGenius uses a deep neural network to embed information into the high-frequency features of an image, making it robust and secure.

This project wraps the powerful PyTorch backend in a **premium, professional-grade Web Dashboard** designed for clarity, ease of use, and visual appeal.

## ✨ Key Features

### 🖥️ Professional Web Interface

- **Cyber-Clean Dashboard**: A modern, dark-themed UI built with Glassmorphism aesthetics.
- **Intelligent Console**: A built-in system log panel that tracks every step of the encoding/decoding process with smart auto-scrolling.
- **Drag & Drop Workflow**: Seamless file handling for both encryption and decryption.
- **Instant Previews**: Visual feedback for all uploaded and generated images.

### 🧠 Advanced AI Core (Stego-GAN)

- **Encoder (Alice)**: Merges the _Cover Image_ and _Secret Image_ into a _Stego Image_.
- **Decoder (Bob)**: Extracts the _Secret Image_ from the _Stego Image_.
- **Critic (Eve)**: A discriminator network used during training to ensure the Stego image looks identical to the original.

### ⚡ Performance

- **Real-Time Inference**: Powered by **FastAPI** and optimized PyTorch models for instant results.
- **Production Ready**: Includes pre-trained checkpoints (`checkpoint_ep50.pth`) for immediate use.

---

## 🛠️ Technology Stack

| Component    | Technology             | Description                                       |
| :----------- | :--------------------- | :------------------------------------------------ |
| **AI Model** | **PyTorch**            | Deep Learning framework for the GAN architecture. |
| **Backend**  | **FastAPI**            | High-performance, async Python web framework.     |
| **Frontend** | **Vanilla JS + CSS3**  | Lightweight, framework-free UI for maximum speed. |
| **Styling**  | **CSS Grid & Flexbox** | Responsive layout with custom animations.         |
| **Server**   | **Uvicorn**            | Lightning-fast ASGI server implementation.        |

---

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

- Python 3.8+
- pip (Python Package Manager)
- A GPU is recommended for training, but CPU works fine for inference (running the app).

### Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/stego-genius.git
    cd stego-genius
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### ▶️ Running the Application

1.  **Start the Server**

    ```bash
    uvicorn app:app --reload
    ```

2.  **Access the Dashboard**
    Open your browser and navigate to:
    `http://127.0.0.1:8000`

---

## 🎮 User Guide

### 1. Encryption (Hiding Data)

1.  Navigate to the **Encrypt Data** tab.
2.  Upload a **Cover Image** (the public carrier).
3.  Upload a **Secret Image** (the data you want to hide).
4.  Click **Encrypt & Hide**.
5.  The system will generate a **Stego Image**. Download it!

### 2. Decryption (Revealing Data)

1.  Navigate to the **Decrypt Data** tab.
2.  Upload the **Stego Image** you generated earlier.
3.  Click **Reveal Secret**.
4.  The system will extract and display the hidden **Secret Image**.

---

## 📂 Project Structure

```
file-structure
├── app.py              # FastAPI Backend Server
├── deep_stego.py       # PyTorch Model Definitions (Stego-GAN)
├── inference.py        # Logic for running the model
├── requirements.txt    # Project Dependencies
├── checkpoints/        # Saved Model Weights
│   └── checkpoint_ep50.pth
├── public/             # Frontend Assets
│   ├── index.html      # Main Dashboard
│   └── preview.png     # (Optional) Screenshot of the UI
└── readme.md           # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> _Developed with ❤️ by Fahad Asfaque_
