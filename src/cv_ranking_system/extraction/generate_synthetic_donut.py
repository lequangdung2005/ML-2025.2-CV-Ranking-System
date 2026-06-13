import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from tqdm import tqdm
import random

class SyntheticCVGenerator:
    def __init__(self, output_dir: str = "./data/donut_dataset", image_size=(960, 1280)):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

    def _wrap_text(self, text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list:
        words = str(text).split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def render_cv_image(self, text: str, output_path: Path) -> bool:
        image = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            header_font = ImageFont.truetype("arial.ttf", 26)
        except IOError:
            font = ImageFont.load_default()
            header_font = font

        header_color = random.choice(["#1a365d", "#2d3748", "#2c5282"])
        draw.rectangle([0, 0, self.image_size[0], 60], fill=header_color)
        draw.text((40, 15), "RESUME / CURRICULUM VITAE", font=header_font, fill="white")
        
        margin = 50
        max_width = self.image_size[0] - (2 * margin)
        paragraphs = str(text).split('\n')
        y_pointer = 90
        
        for para in paragraphs:
            if not para.strip():
                y_pointer += 10
                continue
            lines = self._wrap_text(para, font, max_width, draw)
            for line in lines:
                if y_pointer > self.image_size[1] - margin:
                    break
                draw.text((margin, y_pointer), line, font=font, fill="#2d3748")
                y_pointer += 24
            y_pointer += 8
        image.save(output_path, "PNG")
        return True

    def process_and_convert(self, dataset_name: str = "cnamuangtoun/resume-job-description-fit", max_samples: int = 50):
        print(f"🔄 Loading source text dataset: {dataset_name}...")
        raw_dataset = load_dataset(dataset_name)

        splits = ['train']
        if 'validation' in raw_dataset:
            splits.append('validation')
            
        for split in splits:
            current_split_data = raw_dataset[split]
            target_dir = self.train_dir if split == 'train' else self.val_dir
            metadata_file = target_dir / "metadata.jsonl"
            
            print(f"📦 Generating synthetic images for split: [{split}]...")
            # In ra danh sách cột để debug
            print(f"🔍 Dataset columns found: {list(current_split_data[0].keys())}")
            
            generated_count = 0
            
            with open(metadata_file, "w", encoding="utf-8") as f_meta:
                for idx, sample in enumerate(tqdm(current_split_data)):
                    resume_text = ""
                    
                    # 💡 TÌM KIẾM THÔNG MINH: Tự động quét tất cả các cột, lấy cột chứa chuỗi dài nhất làm CV
                    for key, value in sample.items():
                        if value is not None:
                            text_val = str(value)
                            if len(text_val) > len(resume_text):
                                resume_text = text_val
                                
                    # Nếu độ dài CV vẫn quá ngắn (< 100 ký tự) thì bỏ qua
                    if len(resume_text) < 100:
                        continue
                    
                    img_filename = f"synthetic_cv_{generated_count:05d}.png"
                    img_path = target_dir / img_filename
                    
                    # Bước 1: Render ảnh CV
                    self.render_cv_image(resume_text, img_path)
                    
                    # Bước 2: Tạo Jsonl
                    gt_parse_dict = {
                        "text_content": resume_text[:400] + "..." 
                    }
                    ground_truth_str = json.dumps({"gt_parse": gt_parse_dict}, ensure_ascii=False)
                    
                    line_entry = {
                        "file_name": img_filename,
                        "ground_truth": ground_truth_str
                    }
                    f_meta.write(json.dumps(line_entry, ensure_ascii=False) + "\n")
                    
                    f_meta.flush() 
                    os.fsync(f_meta.fileno())
                    
                    generated_count += 1
                    # Dừng khi đủ số lượng ảnh
                    if generated_count >= max_samples:
                        break
                        
            print(f"✅ Created {generated_count} images and metadata entries for {split} split.")

if __name__ == "__main__":
    generator = SyntheticCVGenerator(output_dir="./data/donut_dataset")
    generator.process_and_convert()