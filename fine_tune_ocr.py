import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AdamW, get_scheduler
from card_text_dataset import CardTextDataset
from tqdm import tqdm

# ------------------------------
# Configurations
# ------------------------------
annotations_file = 'roi_annotations.csv'  # CSV file with ROI filenames and ground truth texts
roi_output_dir = './roi_images/'          # Directory containing the ROI images
num_epochs = 5                            # Number of epochs to train
batch_size = 8                            # Batch size for training
learning_rate = 5e-5                      # Learning rate for optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Fine-tuning Function
# ------------------------------
def fine_tune_trocr():
    print("Starting TrOCR fine-tuning...")

    # Load processor and model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    trocr_model.to(device)

    # Set special tokens
    trocr_model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    trocr_model.config.pad_token_id = processor.tokenizer.pad_token_id
    trocr_model.config.eos_token_id = processor.tokenizer.eos_token_id
    trocr_model.config.vocab_size = trocr_model.config.decoder.vocab_size

    # (Optional) Set sequence lengths
    trocr_model.config.max_length = 128  # Adjust based on your data
    trocr_model.config.min_length = 1

    # (Optional) Set num_beams for beam search decoding
    trocr_model.config.num_beams = 4

    # Ensure the model is in training mode
    trocr_model.train()

    # Create dataset and dataloader
    dataset = CardTextDataset(annotations_file, roi_output_dir, processor, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer and scheduler
    optimizer = AdamW(trocr_model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        for batch in tqdm(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = trocr_model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    trocr_model.save_pretrained("fine_tuned_trocr")
    processor.save_pretrained("fine_tuned_trocr")
    print("TrOCR fine-tuning completed and model saved.")

# ------------------------------
# Run the Fine-tuning
# ------------------------------
if __name__ == "__main__":
    fine_tune_trocr()
