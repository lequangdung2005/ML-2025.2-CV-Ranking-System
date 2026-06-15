import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import random

class SyntheticCVGenerator:
    def __init__(self, output_dir: str = None):
        # Tự động xác định thư mục gốc của dự án (Project Root)
        self.project_root = Path(__file__).resolve().parents[3]
        
        if output_dir is None:
            self.output_dir = self.project_root / "data" / "donut_dataset"
        else:
            self.output_dir = Path(output_dir)
            
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        # Cố định kích thước chuẩn của trang CV dọc
        self.img_w = 960
        self.img_h = 1280

    def _wrap_text(self, text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list:
        """Hàm tự động ngắt dòng khi chữ chạm viền trang."""
        words = str(text).split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    # =========================================================================
    # BƯỚC 1: ĐỌC DATA GỐC -> TRÍCH XUẤT CHUẨN -> TẠO METADATA.JSONL
    # =========================================================================
    def create_clean_metadata(self, raw_json_path: str = None, max_samples: int = 150):
        """Đọc file raw JSON, Cắt bớt phần dài, Mượn chéo phần thiếu -> Lưu metadata.jsonl"""
        if raw_json_path is None:
            raw_json_path = self.project_root / "data" / "resume_ner" / "train.json"
            
        print(f"📦 [BƯỚC 1] Đang xử lý, augmentation và bóc tách dữ liệu từ: {raw_json_path}...")
        
        if not os.path.exists(raw_json_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại {raw_json_path}")

        raw_data = []
        with open(raw_json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                raw_data = json.loads(content)
            else:
                for line in content.split('\n'):
                    if line.strip():
                        raw_data.append(json.loads(line))
        
        random.seed(42)
        random.shuffle(raw_data)
        raw_data = raw_data[:max_samples]
        
        # --- GIAI ĐOẠN 1: XÂY DỰNG "KHO CHỨA" (GLOBAL POOL) ĐỂ MƯỢN CHÉO ---
        global_pool = {
            "designation": [], "skill": [], "company": [], 
            "education": [], "location": [], "email": []
        }
        
        for sample in raw_data:
            text = sample.get("text", "")
            for ann in sample.get("annotations", []):
                start, end, label = ann[0], ann[1], ann[2]
                entity_text = " ".join(text[start:end].strip().split())
                label_name = label.lower().replace(" ", "_")
                
                if len(entity_text) >= 2 and label_name in global_pool:
                    if entity_text not in global_pool[label_name]:
                        global_pool[label_name].append(entity_text)

        # --- GIAI ĐOẠN 2: TẠO METADATA VÀ AUGMENTATION ---
        split_idx = int(len(raw_data) * 0.9)
        splits = {'train': raw_data[:split_idx], 'validation': raw_data[split_idx:]}
        
        for split_name, split_data in splits.items():
            target_dir = self.train_dir if split_name == 'train' else self.val_dir
            metadata_file = target_dir / "metadata.jsonl"
            
            print(f"📝 Đang ghi file cấu trúc metadata.jsonl cho tập: [{split_name}]")
            generated_count = 0
            
            with open(metadata_file, "w", encoding="utf-8") as f_meta:
                for sample in split_data:
                    resume_text = sample.get("text", "")
                    annotations = sample.get("annotations", [])
                    
                    if len(resume_text) < 100 or not annotations:
                        continue
                    
                    # 1. Thu thập thực thể gốc của CV hiện tại
                    temp_dict = {}
                    for ann in annotations:
                        start_idx, end_idx, label = ann[0], ann[1], ann[2]
                        entity_text = " ".join(resume_text[start_idx:end_idx].strip().split())
                        
                        if len(entity_text) < 2 or entity_text.lower() in ["curriculum vitae", "resume", "page", "status", "gender"]:
                            continue
                            
                        label_name = label.lower().replace(" ", "_")
                        if label_name not in temp_dict:
                            temp_dict[label_name] = []
                        if entity_text not in temp_dict[label_name]:
                            temp_dict[label_name].append(entity_text)
                    
                    # 2. LOGIC CẮT BỚT (TRUNCATE) VÀ MƯỢN CHÉO (BORROW)
                    # Giới hạn số lượng tối đa để không bị tràn trang
                    max_limits = {
                        "skill": 24,       # Tối đa 24 kỹ năng
                        "designation": 3,  # Tối đa 3 chức danh
                        "company": 4,      # Tối đa 4 công ty
                        "education": 3     # Tối đa 3 trường/bằng cấp
                    }
                    
                    # Các trường bắt buộc phải có để CV trông thực tế
                    essential_keys = ["designation", "skill", "company", "education"]
                    
                    for key in essential_keys:
                        # Nếu bị thiếu hụt -> Mượn ngẫu nhiên từ Global Pool
                        if key not in temp_dict or not temp_dict[key]:
                            # Mượn ngẫu nhiên từ 1 đến max_limits của trường đó
                            borrow_count = random.randint(1, max_limits.get(key, 3))
                            if len(global_pool[key]) >= borrow_count:
                                temp_dict[key] = random.sample(global_pool[key], borrow_count)
                            else:
                                temp_dict[key] = global_pool[key].copy()
                        
                        # Nếu quá dài -> Cắt bớt phần thừa
                        elif len(temp_dict[key]) > max_limits[key]:
                            # Giữ lại ngẫu nhiên thay vì chỉ cắt phần đuôi để tăng độ đa dạng
                            temp_dict[key] = random.sample(temp_dict[key], max_limits[key])

                    if not temp_dict:
                        continue

                    # 3. Chuẩn bị ground_truth cho Donut
                    gt_parse = {k: "; ".join(v) for k, v in temp_dict.items()}
                    
                    # 4. TỰ ĐỘNG DỰNG TEXT CV CHUẨN
                    structured_text_lines = []
                    
                    name_list = temp_dict.get("person", ["PROFESSIONAL CANDIDATE"])
                    structured_text_lines.append(f"HEADER_NAME: {name_list[0].upper()}")
                    
                    contact_info = []
                    emails = temp_dict.get("email", [])
                    locations = temp_dict.get("location", [])
                    
                    if not emails and global_pool["email"]:
                        emails = [random.choice(global_pool["email"])]
                    if not locations and global_pool["location"]:
                        locations = [random.choice(global_pool["location"])]
                        
                    if emails: contact_info.append(emails[0])
                    if locations: contact_info.append(locations[0])
                    
                    if contact_info:
                        structured_text_lines.append(f"HEADER_CONTACT: {'  |  '.join(contact_info)}")
                    
                    # Thứ tự các mục lớn
                    sections_order = [
                        ("designation", "PROFESSIONAL ROLES & OBJECTIVES"),
                        ("skill", "CORE COMPETENCIES & TECHNICAL SKILLS"),
                        ("company", "EMPLOYMENT HISTORY"),
                        ("education", "ACADEMIC & EDUCATION BACKGROUND")
                    ]
                    
                    for key, section_title in sections_order:
                        if key in temp_dict and temp_dict[key]:
                            structured_text_lines.append(f"SECTION_TITLE: {section_title}")
                            
                            items = temp_dict[key]
                            if key == "skill":
                                # Gom 6-8 skill lại thành 1 gạch đầu dòng
                                chunk_size = 6
                                for i in range(0, len(items), chunk_size):
                                    chunk = items[i:i+chunk_size]
                                    structured_text_lines.append(f"BULLET: {', '.join(chunk)}")
                            else:
                                for item in items:
                                    structured_text_lines.append(f"BULLET: {item}")
                                
                    # Các trường rác còn lại
                    for key, values in temp_dict.items():
                        if key not in ["person", "email", "location", "designation", "skill", "company", "education"]:
                            # Tránh in những list rác quá dài
                            if len(values) > 10:
                                values = random.sample(values, 10)
                                
                            structured_text_lines.append(f"SECTION_TITLE: ADDITIONAL {key.upper()}")
                            chunk_size = 4
                            for i in range(0, len(values), chunk_size):
                                chunk = values[i:i+chunk_size]
                                structured_text_lines.append(f"BULLET: {', '.join(chunk)}")

                    img_filename = f"synthetic_cv_{split_name}_{generated_count:05d}.png"
                    
                    line_entry = {
                        "file_name": img_filename,
                        "ground_truth": json.dumps({"gt_parse": gt_parse}, ensure_ascii=False),
                        "text_render": "\n".join(structured_text_lines)
                    }
                    
                    f_meta.write(json.dumps(line_entry, ensure_ascii=False) + "\n")
                    generated_count += 1
                    
            print(f"✅ Đã tạo thành công file metadata.jsonl sạch với {generated_count} mẫu cho [{split_name}].")
            
    # =========================================================================
    # BƯỚC 2: ĐỌC METADATA.JSONL -> ĐỒNG BỘ VÀ VẼ ẢNH CV
    # =========================================================================
    def generate_images_from_metadata(self):
        """Đọc trực tiếp từ file metadata.jsonl đã tạo ở Bước 1 để vẽ ảnh chuẩn 100%"""
        print("🎨 [BƯỚC 2] Đang tiến hành đọc file metadata để dựng ảnh CV chuẩn hóa...")
        
        for split_name in ['train', 'validation']:
            target_dir = self.train_dir if split_name == 'train' else self.val_dir
            metadata_file = target_dir / "metadata.jsonl"
            
            if not metadata_file.exists():
                print(f"⚠️ Không tìm thấy file metadata tại {metadata_file}, bỏ qua tập {split_name}.")
                continue
                
            # Đọc danh sách cấu trúc để chạy Tqdm
            with open(metadata_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                
            print(f"🖼️ Dựng {len(lines)} ảnh CV cho tập [{split_name}]...")
            
            for item in tqdm(lines):
                img_filename = item["file_name"]
                text_render = item["text_render"]
                output_path = target_dir / img_filename
                
                # Thực hiện vẽ ảnh từ cấu trúc chữ sạch
                self._render_single_cv_image(text_render, output_path)
                
        print("🎉 Toàn bộ pipeline Tạo Metadata và Dựng ảnh đã hoàn tất hoàn hảo!")

    def _render_single_cv_image(self, structured_text: str, output_path: Path):
        """Hàm lõi dựng đồ họa cho trang giấy CV"""
        image = Image.new("RGB", (self.img_w, self.img_h), "#F8FAFC") 
        draw = ImageDraw.Draw(image)

        try:
            font_name = "calibri.ttf"  
            font_title = ImageFont.truetype(font_name, 26)
            font_subtitle = ImageFont.truetype(font_name, 13)
            font_header = ImageFont.truetype(font_name, 16)
            font_body = ImageFont.truetype(font_name, 13)
        except IOError:
            font_title = font_subtitle = font_header = font_body = ImageFont.load_default()

        COLOR_PRIMARY = "#1E3A8A"   
        COLOR_TEXT_MAIN = "#1E293B" 
        COLOR_TEXT_MUTED = "#64748B"
        COLOR_LINE = "#CBD5E1"      

        margin_left = 75
        margin_right = 75
        y_pointer = 60

        lines = structured_text.split('\n')
        
        for line in lines:
            if y_pointer > self.img_h - 60:
                break
                
            if line.startswith("HEADER_NAME:"):
                name_text = line.replace("HEADER_NAME:", "").strip()
                name_bbox = draw.textbbox((0, 0), name_text, font=font_title)
                name_w = name_bbox[2] - name_bbox[0]
                draw.text(((self.img_w - name_w) // 2, y_pointer), name_text, font=font_title, fill=COLOR_PRIMARY)
                y_pointer += 45
                
            elif line.startswith("HEADER_CONTACT:"):
                contact_text = line.replace("HEADER_CONTACT:", "").strip()
                info_bbox = draw.textbbox((0, 0), contact_text, font=font_subtitle)
                info_w = info_bbox[2] - info_bbox[0]
                draw.text(((self.img_w - info_w) // 2, y_pointer), contact_text, font=font_subtitle, fill=COLOR_TEXT_MUTED)
                y_pointer += 50
                
            elif line.startswith("SECTION_TITLE:"):
                title_text = line.replace("SECTION_TITLE:", "").strip()
                y_pointer += 15 
                draw.text((margin_left, y_pointer), title_text, font=font_header, fill=COLOR_PRIMARY)
                y_pointer += 22
                draw.line([(margin_left, y_pointer), (self.img_w - margin_right, y_pointer)], fill=COLOR_LINE, width=1)
                y_pointer += 15
                
            elif line.startswith("BULLET:"):
                bullet_content = "• " + line.replace("BULLET:", "").strip()
                max_txt_w = self.img_w - margin_left - margin_right - 20
                wrapped_lines = self._wrap_text(bullet_content, font_body, max_txt_w, draw)
                
                for idx, w_line in enumerate(wrapped_lines):
                    if y_pointer > self.img_h - 60: 
                        break
                    current_indent = margin_left if idx == 0 else margin_left + 15
                    draw.text((current_indent, y_pointer), w_line, font=font_body, fill=COLOR_TEXT_MAIN)
                    y_pointer += 22
                y_pointer += 4

        image.save(output_path, "PNG")


if __name__ == "__main__":
    # Khởi chạy pipeline 2 bước quy chuẩn
    generator = SyntheticCVGenerator()
    
    # Bước 1: Trích xuất sạch -> Tạo file metadata.jsonl trước
    generator.create_clean_metadata(max_samples=150)
    
    # Bước 2: Đọc file metadata.jsonl -> Dựng ảnh CV tương ứng theo cấu trúc chuẩn
    generator.generate_images_from_metadata()