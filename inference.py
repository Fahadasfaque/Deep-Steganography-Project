import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from deep_stego import PrepNet, HidingNet, RevealNet  # Importing models from main script

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models from {args.checkpoint}...")
    
    # Load Models
    prep_net = PrepNet().to(device)
    hiding_net = HidingNet().to(device)
    reveal_net = RevealNet().to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    prep_net.load_state_dict(checkpoint['prep_net'])
    hiding_net.load_state_dict(checkpoint['hiding_net'])
    reveal_net.load_state_dict(checkpoint['reveal_net'])
    
    prep_net.eval()
    hiding_net.eval()
    reveal_net.eval()
    
    # Process Images
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])
    
    print(f"Processing Cover: {args.cover}")
    print(f"Processing Secret: {args.secret}")
    
    cover = Image.open(args.cover).convert('RGB')
    secret = Image.open(args.secret).convert('RGB')
    
    cover_tensor = transform(cover).unsqueeze(0).to(device)
    secret_tensor = transform(secret).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        # Hide
        prep_secret = prep_net(secret_tensor)
        stego = hiding_net(cover_tensor, prep_secret)
        
        # Reveal
        revealed = reveal_net(stego)
        
        # Residual (Diff)
        residual = (stego - cover_tensor).abs() * 10  # Amplify diff for visibility
    
    # Save Outputs
    # Stego
    stego_img = transforms.ToPILImage()(stego.squeeze(0).cpu().clamp(0, 1))
    stego_img.save(args.output_stego)
    
    # Revealed
    revealed_img = transforms.ToPILImage()(revealed.squeeze(0).cpu().clamp(0, 1))
    revealed_img.save(args.output_reveal)
    
    print(f"Saved Stego Image to: {args.output_stego}")
    print(f"Saved Revealed Secret to: {args.output_reveal}")

    # Plot
    if args.plot:
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        ax[0].imshow(cover)
        ax[0].set_title("Cover")
        ax[1].imshow(secret)
        ax[1].set_title("Secret")
        ax[2].imshow(stego_img)
        ax[2].set_title("Stego (Ours)")
        ax[3].imshow(transforms.ToPILImage()(residual.squeeze(0).cpu().clamp(0, 1)))
        ax[3].set_title("Residual (x10)")
        ax[4].imshow(revealed_img)
        ax[4].set_title("Revealed")
        for a in ax: a.axis('off')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stego-GAN Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--cover', type=str, required=True, help='Path to Cover image')
    parser.add_argument('--secret', type=str, required=True, help='Path to Secret image')
    parser.add_argument('--output_stego', type=str, default='stego_out.png')
    parser.add_argument('--output_reveal', type=str, default='reveal_out.png')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--plot', action='store_true', help='Show plot')
    
    args = parser.parse_args()
    inference(args)
