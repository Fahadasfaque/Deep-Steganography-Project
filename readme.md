# 🕵️‍♂️ Deep Steganography Experience

![Project Banner](public/preview.png)

> _An advanced Deep Learning application that hides secrets inside images using Steganographic GANs, wrapped in a premium Glassmorphism web interface._

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-High_Performance-green?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![HTML5](https://img.shields.io/badge/HTML5-Modern-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)
[![CSS3](https://img.shields.io/badge/CSS3-Glassmorphism-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)

---

## 🌟 Overview

Welcome to **Deep Steganography**. This project isn't just a security tool; it's a **state-of-the-art AI system** accessible through a beautiful web dashboard. It allows you to encipher secret data into ordinary images, making the secret invisible to the naked eye. Perfect for secure communication or digital watermarking demonstrations.

## ✨ Key Features

- **🧠 Deep Learning Core**: Uses a custom **U-Net + GAN** architecture (Alice, Bob, and Eve networks) to learn perfect hiding strategies.
- **👁️ Invisible Secrecy**: The "Stego Image" looks identical to the original Cover Image.
- **🖥️ Glassmorphism UI**: A premium, modern web interface designed with **translucency, blur effects, and smooth animations**.
- **🚀 Instant Results**: Powered by **FastAPI** for real-time inference.
- **🖱️ Drag & Drop**: Intuitive workflow—just drop your images to hide or reveal secrets.
- **📱 Responsive Design**: Works beautifully on desktops and large screens.

---

## 🛠️ Tech Stack & Libraries

This project uses modern AI and Web technologies:

| Category     | Technology       | Usage                                                                   |
| :----------- | :--------------- | :---------------------------------------------------------------------- |
| **AI Core**  | **PyTorch**      | The brain. Handles the Neural Networks (PrepNet, HidingNet, RevealNet). |
| **Backend**  | **FastAPI**      | High-performance API server to run the Python models.                   |
| **Server**   | **Uvicorn**      | Lightning-fast ASI server implementation.                               |
| **Frontend** | **Vanilla JS**   | Lightweight, fast logic for the UI without framework bloat.             |
| **Styling**  | **CSS3 (Glass)** | Custom "Glassmorphism" aesthetic with blurred backgrounds.              |
| **Data**     | **Pillow (PIL)** | Advanced image processing and manipulation.                             |

---

## 🚀 Getting Started

Clone the repository and start the magic in seconds.

### Prerequisites

- Python 3.8 or higher
- A love for secrets 🤫

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Start the server with hot-reload
uvicorn app:app --reload
```

Visit `http://localhost:8000` to access the interface.

---

## �️ Workflow & Architecture

The project follows a "Client-Server-AI" architecture:

1.  **The User Interface**:
    - Users interact with the **HTML/CSS/JS** frontend.
    - Images are uploaded via Drag & Drop.
2.  **The API Layer (FastAPI)**:
    - Receives images at `/api/hide` or `/api/reveal`.
    - Converts images to PyTorch Tensors.
3.  **The AI Engine (Deep Stego)**:
    - **Encoder (Alice)**: Merges the Cover and Secret into a Stego image.
    - **Decoder (Bob)**: Extracts the Secret from the Stego image.
4.  **Response**:
    - The processed image is streamed back to the browser instantly.

---

## � Deployment Ready

This application is production-ready. The model checkpoint (`checkpoint_ep50.pth`) is pre-loaded, and the server is optimized for inference.

---

## 🎨 Credits

Designed and developed with ❤️ by **Fahad Asfaque**.

_Inspired by the concept of "Security through Obscurity" in the AI era._

---

_Note: This project is for educational and showcase purposes._
