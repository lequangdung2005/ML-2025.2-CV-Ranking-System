import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import random

# =========================================================================
# CẤU HÌNH CÁC BẢNG MÀU CHUYÊN NGHIỆP CÓ ĐỘ PHẢN CHIẾU CAO
# =========================================================================
THEMES = [
    {"primary": "#1E3A8A", "text_main": "#1E293B", "text_muted": "#475569", "bg": "#FFFFFF", "sidebar_bg": "#F8FAFC", "line": "#CBD5E1"}, 
    {"primary": "#064E3B", "text_main": "#0F172A", "text_muted": "#4B5563", "bg": "#FFFFFF", "sidebar_bg": "#F0FDF4", "line": "#BBF7D0"}, 
    {"primary": "#18181B", "text_main": "#27272A", "text_muted": "#71717A", "bg": "#FFFFFF", "sidebar_bg": "#FAFAFA", "line": "#E4E4E7"}, 
    {"primary": "#991B1B", "text_main": "#1C1917", "text_muted": "#78716C", "bg": "#FFFDFA", "sidebar_bg": "#FDF6ED", "line": "#FED7AA"}
]
def format_donut_gt(temp_dict):
    gt_string = ""
    for key, values in temp_dict.items():
        # values là một list các string, nối lại bằng dấu "; "
        val_str = "; ".join(values)
        gt_string += f"<s_{key}>{val_str}</s_{key}>"
    return gt_string

class SyntheticCVGenerator:
    def __init__(self, output_dir: str = None):
        self.project_root = Path(__file__).resolve().parents[3]
        
        if output_dir is None:
            self.output_dir = self.project_root / "data" / "donut_dataset"
        else:
            self.output_dir = Path(output_dir)
            
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        self.img_w = 960
        self.img_h = 1280

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
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    # =========================================================================
    # BƯỚC 1: LÀM GIÀU DỮ LIỆU KỊCH TRẦN (MAXIMUM DENSITY SCHEMA)
    # =========================================================================
    def create_clean_metadata(self, raw_json_path: str = None, max_samples: int = 150):
        if raw_json_path is None:
            raw_json_path = self.project_root / "data" / "resume_ner" / "train.json"
            
        print(f"📦 [BƯỚC 1] Đang thực hiện tối đa hóa mật độ chữ và phân mục mở rộng...")
        
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
        
        global_pool = {"designation": [], "skill": [], "company": [], "education": [], "location": [], "email": []}
        for sample in raw_data:
            text = sample.get("text", "")
            for ann in sample.get("annotations", []):
                start, end, label = ann[0], ann[1], ann[2]
                entity_text = " ".join(text[start:end].strip().split())
                label_name = label.lower().replace(" ", "_")
                if len(entity_text) >= 2 and label_name in global_pool:
                    if entity_text not in global_pool[label_name]:
                        global_pool[label_name].append(entity_text)

        # =========================================================================
        # NGÂN HÀNG DỮ LIỆU SIÊU MỞ RỘNG (LEXICAL DIVERSITY POOLS)
        # =========================================================================
        
        # 1. Câu mô tả hành động (Job Descriptions) - Đa lĩnh vực, cấu trúc dài phức tạp
        pool_job_descs = [
            "Spearheaded core architectural planning, microservices migrations, and end-to-end development workflows to drive major milestones.",
            "Optimized distributed systems throughput, diagnosed memory bottlenecks, and improved server response latency by over 35%.",
            "Designed scalable relational database structures, implemented strict query caching layers, and orchestrated secure OAuth2 workflows.",
            "Mentored team members through rigorous code reviews, fostered continuous deployment integrations, and enhanced code coverage to 92%.",
            "Collaborated directly with engineering leaders to map technical debt and execute high-priority refactoring modules efficiently.",
            "Managed real-time infrastructure event streaming pipes, mitigating continuous packet delivery issues under high-load production hours.",
            "Orchestrated cloud infrastructure deployment using Infrastructure as Code (IaC) principles, reducing provisioning time by 50%.",
            "Enforced strict security protocols, conducted vulnerability assessments, and patched high-risk authorization flaws across legacy web portals.",
            "Built data ingestion pipelines processing multi-terabyte datasets daily, ensuring high-availability data delivery to analytics clusters.",
            "Developed responsive user interfaces leveraging state-of-the-art frontend patterns, achieving a 98% Lighthouse performance score.",
            "Implemented end-to-end automated testing suites incorporating integration, regression, and stress testing models under CI/CD gates.",
            "Led cross-functional design sprints to translate ambiguous client specifications into production-ready system design documents.",
            "Architected secure financial transaction modules complying with strict compliance criteria and encryption standard requirements.",
            "Refactored legacy monolith applications into modular domain-driven services, decreasing cross-team deployment blockers dramatically.",
            "Monitored multi-cluster Kubernetes deployments, optimizing compute resource allocation and slashing monthly cloud expenditures by 20%."
        ]
        
        # 2. Dự án kỹ thuật chuyên sâu (Projects) - Đa dạng domain từ Web, AI, IoT đến Blockchain
        pool_projects = [
            {
                "name": "Enterprise Microservices E-Commerce Infrastructure", 
                "tech": "Go, gRPC, Kubernetes, Kafka, Redis", 
                "desc": "Led the development of a decoupled booking engine processing up to 15,000 requests per minute with robust fault tolerance mechanism."
            },
            {
                "name": "Multi-Modal Deep Learning Document Extractor", 
                "tech": "Python, PyTorch, HuggingFace, FastAPI, Docker", 
                "desc": "Built a custom sequence-to-sequence transformer model optimized for parsing multi-lingual structured forms with 96.4% layout accuracy."
            },
            {
                "name": "Distributed Ledger Financial Reconciliation System", 
                "tech": "Java, Spring Boot, PostgreSQL, Docker, AWS", 
                "desc": "Engineered an automated data matching core capable of reconciling million-row statements across multi-bank gateways within seconds."
            },
            {
                "name": "Real-Time Event Analytics Stream Engine", 
                "tech": "Scala, Apache Flink, Redis Cluster, Prometheus", 
                "desc": "Implemented a zero-egress telemetry dashboard tracking client behavior pipelines across web nodes with sub-second visual refresh cycles."
            },
            {
                "name": "Predictive Maintenance IoT Telemetry Core",
                "tech": "Python, MQTT, TensorFlow, TimescaleDB, Grafana",
                "desc": "Deployed anomalous pattern detector algorithms processing machinery sensor streams, forecasting hardware degradation 48 hours in advance."
            },
            {
                "name": "Decentralized Identity Verification Protocol",
                "tech": "Solidity, Ethereum, Node.js, Web3.js, IPFS",
                "desc": "Created a zero-knowledge proof compliance gate for secure digital identity validation without disclosing underlying demographic records."
            },
            {
                "name": "Cloud-Native Distributed Log Aggregation Daemon",
                "tech": "Rust, WebAssembly, ClickHouse, Vector, GCP",
                "desc": "Engineered a low-footprint scraping daemon processing raw container outputs with efficient parsing speeds and optimal memory constraints."
            },
            {
                "name": "Automated Healthcare Telemetry Orchestrator",
                "tech": "Python, Django, AWS IoT Core, PostgreSQL, React",
                "desc": "Designed HIPAA-compliant data transit system streaming patient vitals securely from remote medical devices to active monitoring nodes."
            },
            {
                "name": "High-Frequency Algorithmic Order Matching Engine",
                "tech": "C++, ZeroMQ, Redis, Linux Kernel Tuning",
                "desc": "Implemented a customized order execution gateway with sub-millisecond network latency constraints for high-frequency algorithmic equity pairs."
            }
        ]
                
        split_idx = int(len(raw_data) * 0.9)
        splits = {'train': raw_data[:split_idx], 'validation': raw_data[split_idx:]}
        
        for split_name, split_data in splits.items():
            target_dir = self.train_dir if split_name == 'train' else self.val_dir
            metadata_file = target_dir / "metadata.jsonl"
            
            generated_count = 0
            with open(metadata_file, "w", encoding="utf-8") as f_meta:
                for sample in split_data:
                    resume_text = sample.get("text", "")
                    annotations = sample.get("annotations", [])
                    if len(resume_text) < 100 or not annotations:
                        continue
                    
                    temp_dict = {}
                    for ann in annotations:
                        start_idx, end_idx, label = ann[0], ann[1], ann[2]
                        entity_text = " ".join(resume_text[start_idx:end_idx].strip().split())
                        if len(entity_text) < 2 or entity_text.lower() in ["curriculum vitae", "resume", "page", "status", "gender"]:
                            continue
                        label_name = label.lower().replace(" ", "_")
                        if label_name not in temp_dict: temp_dict[label_name] = []
                        if entity_text not in temp_dict[label_name]: temp_dict[label_name].append(entity_text)
                    
                    # ĐẨY MẠNH NGƯỠNG TỐI THIỂU (MIN-BOUNDS MULTIPLIERS) TOÀN BỘ CÁC TRƯỜNG CƠ BẢN
                    limits = {
                        "skill": {"min": 20, "max": 24},       # Gấp đôi lượng skill
                        "company": {"min": 3, "max": 4},     # Ép buộc 3-4 công ty lớn
                        "designation": {"min": 3, "max": 4}, 
                        "education": {"min": 2, "max": 2}    # Ít nhất 2 trường/bằng cấp
                    }
                    
                    for key, bound in limits.items():
                        if key not in temp_dict: temp_dict[key] = []
                        current_items = temp_dict[key]
                        if len(current_items) < bound["min"]:
                            target_count = random.randint(bound["min"], bound["max"])
                            needed = target_count - len(current_items)
                            available = [item for item in global_pool[key] if item not in current_items]
                            temp_dict[key].extend(random.sample(available, min(needed, len(available))) if available else global_pool[key][:needed])
                        elif len(current_items) > bound["max"]:
                            temp_dict[key] = random.sample(current_items, bound["max"])

                    # TIÊM THÊM 4 PHÂN MỤC NÂNG CAO ĐỂ TRANG GIẤY FULL KHÔNG TỲ VẾT
                    chosen_projects = random.sample(pool_projects, 3) # Luôn lấy hẳn 3 dự án lớn
                    temp_dict["projects"] = [f"{p['name']} ({p['tech']})" for p in chosen_projects]
                    project_descriptions = [p['desc'] for p in chosen_projects]
                    
                    if not temp_dict: continue

                    # ĐỒNG BỘ TOÀN DIỆN GROUND TRUTH VỚI METADATA
                    gt_parse = {k: "; ".join(v) for k, v in temp_dict.items()}
                    
                    # --- XÂY DỰNG CHUỖI TEXT RENDER TOÀN DIỆN ---
                    structured_text_lines = []
                    
                    # Header
                    name_list = temp_dict.get("person", ["EXECUTIVE CANDIDATE"])
                    structured_text_lines.append(f"HEADER_NAME: {name_list[0].upper()}")
                    contact_info = []
                    emails = temp_dict.get("email", [])
                    locations = temp_dict.get("location", [])
                    if not emails and global_pool["email"]: emails = [random.choice(global_pool["email"])]
                    if not locations and global_pool["location"]: locations = [random.choice(global_pool["location"])]
                    if emails: contact_info.append(emails[0])
                    if locations: contact_info.append(locations[0])
                    if contact_info: structured_text_lines.append(f"HEADER_CONTACT: {'  |  '.join(contact_info)}")
                    
                    # Summary
                    structured_text_lines.append("SECTION_TITLE: PROFESSIONAL SUMMARY")
                    structured_text_lines.append("SUMMARY_TEXT: Highly adaptive, results-oriented engineering professional with an extensive track record of delivering scalable web architectures and robust algorithmic infrastructures. Proven capability to analyze code performance, mitigate database bottlenecks, and execute seamless product feature integrations in fast-paced agile development frameworks.")
                    
                    # Skills (Gom 4 skill một dòng)
                    structured_text_lines.append("SECTION_TITLE: CORE COMPETENCIES & TECHNICAL SKILLS")
                    for i in range(0, len(temp_dict["skill"]), 4):
                        structured_text_lines.append(f"BULLET: {', '.join(temp_dict['skill'][i:i+4])}")
                    
                    # Experience (Mỗi nơi tăng lên 3 gạch đầu dòng dài)
                    structured_text_lines.append("SECTION_TITLE: PROFESSIONAL EXPERIENCE")
                    designations = temp_dict.get("designation", ["Senior Software Specialist"])
                    for idx, comp in enumerate(temp_dict["company"]):
                        desig = designations[idx % len(designations)]
                        structured_text_lines.append(f"BULLET_HEADER: {desig.upper()}  |  {comp} ({2018 + idx} - {2021 + idx if idx < 2 else 'Present'})")
                        for desc in random.sample(pool_job_descs, 3): # Luôn ép 3 gạch đầu dòng chi tiết
                            structured_text_lines.append(f"BULLET_DESC: {desc}")
                                
                    # Projects (3 dự án lớn kèm mô tả chi tiết)
                    structured_text_lines.append("SECTION_TITLE: KEY ENGINEERING PROJECTS")
                    for idx, proj_str in enumerate(temp_dict["projects"]):
                        structured_text_lines.append(f"BULLET_HEADER: {proj_str}")
                        structured_text_lines.append(f"BULLET_DESC: {project_descriptions[idx]}")
                                
                    # Education
                    structured_text_lines.append("SECTION_TITLE: ACADEMIC & EDUCATION BACKGROUND")
                    for edu in temp_dict["education"]:
                        structured_text_lines.append(f"BULLET_HEADER: {edu} (Graduated Timeline Study)")
                                            
                    layout_type = random.choice(["single_column", "two_column"])
                    theme_idx = random.randint(0, len(THEMES) - 1)

                    img_filename = f"synthetic_cv_{split_name}_{generated_count:05d}.png"
                    line_entry = {
                        "file_name": img_filename,
                        "ground_truth": format_donut_gt(temp_dict),
                        "text_render": "\n".join(structured_text_lines),
                        "layout_type": layout_type,
                        "theme_idx": theme_idx
                    }
                    f_meta.write(json.dumps(line_entry, ensure_ascii=False) + "\n")
                    generated_count += 1
                    
            print(f"✅ Đã tạo thành công file metadata.jsonl sạch với {generated_count} mẫu cho [{split_name}].")

    # =========================================================================
    # BƯỚC 2: RENDER ẢNH - ĐẢM BẢO CHỮ TRẢI KHÍT TRANG GIẤY
    # =========================================================================
    def generate_images_from_metadata(self):
        print("🎨 [BƯỚC 2] Đang đọc file metadata và tiến hành dựng đa dạng layout hình ảnh...")
        for split_name in ['train', 'validation']:
            target_dir = self.train_dir if split_name == 'train' else self.val_dir
            metadata_file = target_dir / "metadata.jsonl"
            if not metadata_file.exists(): continue
                
            with open(metadata_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                
            print(f"🖼️ Đang sinh {len(lines)} ảnh cho tập [{split_name}]...")
            for item in tqdm(lines):
                self._render_single_cv_image(
                    structured_text=item["text_render"],
                    output_path=target_dir / item["file_name"],
                    layout_type=item.get("layout_type", "single_column"),
                    theme_idx=item.get("theme_idx", 0)
                )

    def _render_single_cv_image(self, structured_text: str, output_path: Path, layout_type: str, theme_idx: int):
        theme = THEMES[theme_idx]
        COLOR_PRIMARY = theme["primary"]
        COLOR_TEXT_MAIN = theme["text_main"]
        COLOR_TEXT_MUTED = theme["text_muted"]
        COLOR_LINE = theme["line"]

        image = Image.new("RGB", (self.img_w, self.img_h), theme["bg"])
        draw = ImageDraw.Draw(image)

        # Định hình độ rộng Sidebar lớn hơn một chút (340px)
        if layout_type == "two_column":
            draw.rectangle([(0, 0), (340, self.img_h)], fill=theme["sidebar_bg"])

        # -------------------------------------------------------------------
        # TĂNG CỠ CHỮ ĐỒNG LOẠT (Tăng trung bình từ 3-7 point)
        # -------------------------------------------------------------------
        try:
            font_title = ImageFont.truetype("liberationb.ttf", 32)        # Cũ: 25 -> Mới: 32
            font_subtitle = ImageFont.truetype("liberation.ttf", 15)      # Cũ: 12 -> Mới: 15
            font_header = ImageFont.truetype("liberationb.ttf", 18)       # Cũ: 14 -> Mới: 18
            font_body = ImageFont.truetype("liberation.ttf", 14)          # Cũ: 12 -> Mới: 14
            font_body_bold = ImageFont.truetype("liberationb.ttf", 14)    # Cũ: 12 -> Mới: 14
        except IOError:
            try:
                font_title = ImageFont.truetype("calibrib.ttf", 32)
                font_subtitle = ImageFont.truetype("calibri.ttf", 15)
                font_header = ImageFont.truetype("calibrib.ttf", 18)
                font_body = ImageFont.truetype("calibri.ttf", 14)
                font_body_bold = ImageFont.truetype("calibrib.ttf", 14)
            except IOError:
                font_title = font_subtitle = font_header = font_body = font_body_bold = ImageFont.load_default()

        lines = structured_text.split('\n')
        header_lines = []
        sections = []
        current_section = None
        
        for line in lines:
            if line.startswith("HEADER_NAME:") or line.startswith("HEADER_CONTACT:"):
                header_lines.append(line)
            elif line.startswith("SECTION_TITLE:"):
                current_section = {"title": line.replace("SECTION_TITLE:", "").strip(), "lines": []}
                sections.append(current_section)
            else:
                if current_section is not None:
                    current_section["lines"].append(line)

        y_pointer = 55

        # Render Header (Tăng các bước nhảy y_pointer để giãn dòng)
        if layout_type == "single_column":
            for line in header_lines:
                txt = line.split(":", 1)[1].strip()
                font = font_title if "NAME" in line else font_subtitle
                color = COLOR_PRIMARY if "NAME" in line else COLOR_TEXT_MUTED
                w = draw.textbbox((0, 0), txt, font=font)[2] - draw.textbbox((0, 0), txt, font=font)[0]
                draw.text(((self.img_w - w) // 2, y_pointer), txt, font=font, fill=color)
                y_pointer += 45  # Cũ: 35 -> Mới: 45
            y_pointer += 15      # Cũ: 10 -> Mới: 15
        else:
            for line in header_lines:
                txt = line.split(":", 1)[1].strip()
                font = font_title if "NAME" in line else font_subtitle
                color = COLOR_PRIMARY if "NAME" in line else COLOR_TEXT_MUTED
                draw.text((375, y_pointer), txt, font=font, fill=color)
                y_pointer += 40  # Cũ: 34 -> Mới: 40
            y_pointer += 20      # Cũ: 15 -> Mới: 20

        # Hàm render một section block
        def draw_section_block(sec, x_start, max_width, y_start):
            y = y_start
            if y > self.img_h - 45: return y
            
            draw.text((x_start, y), sec["title"], font=font_header, fill=COLOR_PRIMARY)
            y += 24  # Cũ: 18 -> Mới: 24 (Cách dòng sau tiêu đề)
            draw.line([(x_start, y), (x_start + max_width, y)], fill=COLOR_LINE, width=1)
            y += 12  # Cũ: 10 -> Mới: 12 (Cách dòng sau đường kẻ)
            
            for l in sec["lines"]:
                if y > self.img_h - 45: break
                
                # Tăng khoảng cách dòng khi in ra (y += ...)
                if l.startswith("SUMMARY_TEXT:"):
                    wrapped = self._wrap_text(l.replace("SUMMARY_TEXT:", "").strip(), font_body, max_width, draw)
                    for wl in wrapped:
                        draw.text((x_start, y), wl, font=font_body, fill=COLOR_TEXT_MAIN)
                        y += 22  # Cũ: 18 -> Mới: 22
                    y += 6       # Cũ: 4 -> Mới: 6
                    
                elif l.startswith("BULLET_HEADER:"):
                    wrapped = self._wrap_text("• " + l.replace("BULLET_HEADER:", "").strip(), font_body_bold, max_width, draw)
                    for wl in wrapped:
                        draw.text((x_start, y), wl, font=font_body_bold, fill=COLOR_TEXT_MAIN)
                        y += 22  # Cũ: 18 -> Mới: 22
                        
                elif l.startswith("BULLET_DESC:"):
                    wrapped = self._wrap_text("- " + l.replace("BULLET_DESC:", "").strip(), font_body, max_width - 15, draw)
                    for wl in wrapped:
                        draw.text((x_start + 15, y), wl, font=font_body, fill=COLOR_TEXT_MUTED)
                        y += 20  # Cũ: 17 -> Mới: 20
                        
                elif l.startswith("BULLET:"):
                    wrapped = self._wrap_text("• " + l.replace("BULLET:", "").strip(), font_body, max_width, draw)
                    for idx, wl in enumerate(wrapped):
                        indent = x_start if idx == 0 else x_start + 12
                        draw.text((indent, y), wl, font=font_body, fill=COLOR_TEXT_MAIN)
                        y += 22  # Cũ: 18 -> Mới: 22
                    y += 6       # Cũ: 4 -> Mới: 6
            
            return y + 16  # Cũ: 14 -> Mới: 16 (Khoảng cách giữa các Section)

        # --- ĐIỀU PHỐI LAYOUT CHẶT CHẼ ĐỂ TRÁNH TRỐNG TRANG GIẤY ---
        if layout_type == "single_column":
            for sec in sections:
                y_pointer = draw_section_block(sec, 75, 810, y_pointer)
        else:
            y_sidebar = 55
            y_main = y_pointer
            
            sidebar_keywords = ["COMPETENCIES", "ACADEMIC", "CERTIFICATIONS", "HONORS", "ADDITIONAL"]
            for sec in sections:
                is_sidebar = any(kw in sec["title"] for kw in sidebar_keywords)
                if is_sidebar:
                    y_sidebar = draw_section_block(sec, 30, 280, y_sidebar)
                else:
                    y_main = draw_section_block(sec, 375, 510, y_main)

        image.save(output_path, "PNG")
        

if __name__ == "__main__":
    generator = SyntheticCVGenerator()
    generator.create_clean_metadata(max_samples=1500)
    generator.generate_images_from_metadata()