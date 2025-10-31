#!/usr/bin/env python3
"""
Translate Vietnamese comments and markdown text to English in Jupyter notebooks.

This script translates:
- Code comments: text after the first unquoted # in Python code
- Markdown cells: Vietnamese text (excluding fenced code blocks and inline code)

It preserves:
- All Python code, imports, and logic
- Execution counts, outputs, and metadata
- Fenced code blocks (``` or ~~~) and inline code (`...`)
"""

import json
import re
import sys
from pathlib import Path

# Import the comprehensive translation dictionary
import os
import sys

# Add the tools directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from translations import TRANSLATIONS
    USE_COMPREHENSIVE = True
except ImportError:
    # Fallback minimal dictionary if translations.py not found
    TRANSLATIONS = {}
    USE_COMPREHENSIVE = False


class NotebookTranslator:
    def __init__(self):
        # Use the comprehensive translation dictionary if available
        if USE_COMPREHENSIVE and TRANSLATIONS:
            self.translation_dict = TRANSLATIONS
        else:
            # Fallback to inline dictionary
            self.translation_dict = {
            # Common phrases and sentences
            'Merge data train từ các file': 'Merge training data from files',
            'Bật memory growth để tránh chiếm trọn VRAM': 'Enable memory growth to avoid occupying all VRAM',
            'log device placement để chắc chắn đang chạy trên GPU': 'log device placement to ensure it is running on the GPU',
            'tuỳ chọn': 'optional',
            'xử lý holiday': 'process holidays',
            'xử lý': 'process',
            'Kiểm tra số event mỗi ngày': 'Check the number of events per day',
            'Lọc ra ngày có nhiều hơn 1 sự kiện': 'Filter days with more than one event',
            'xem thử 10 ngày đầu': 'preview the first 10 days',
            'merge các bảng KHÁC holiday': 'merge the tables EXCEPT the holiday table',
            'để NaN, xử lý sau': 'leave as NaN, process later',
            'Xem chi tiết các event trong ngày': 'View details of events on',
            'merge holiday đúng phạm vi và gộp về một cột/loại': 'merge holidays with correct scope and consolidate into one column/type',
            'thêm 2 cột theo note': 'add 2 columns according to note',
            'Hiện tất cả các cột': 'Show all columns',
            'Không giới hạn độ rộng': 'No width limit',
            'Điền giá trị null bằng giá trị của ngày liền trước': 'Fill null values with values from the previous day',
            'Vì ffill chỉ điền từ trên xuống, nếu dòng đầu tiên bị null thì nó vẫn giữ nguyên.': 'Since ffill only fills from top to bottom, if the first row is null it remains unchanged.',
            'Bạn có thể kết hợp bfill để lấp bằng ngày liền sau:': 'You can combine bfill to fill with the next day:',
            'Chuẩn hoá tên cột để code vẽ dùng chung 1 tên': 'Standardize column names so the plotting code uses a common name',
            'Chuẩn bị dữ liệu theo năm-tháng': 'Prepare data by year-month',
            'đảm bảo đủ 12 tháng mỗi năm': 'ensure 12 months per year',
            'Vẽ grouped bar với cùng 1 tone màu (Blues)': 'Draw grouped bar with the same color tone (Blues)',
            'từ nhạt tới đậm': 'from light to dark',
            'Kiểm tra phân phối': 'Check distribution',
            'Phân phối của các biến số sales, dcoilwtico, transactions, promotion': 'Distribution of numerical variables sales, dcoilwtico, transactions, promotion',
            'log1p cho các biến skew': 'log1p for skewed variables',
            'Phân phối của các biến số sales, dcoilwtico, transactions, promotion sau log1p': 'Distribution of numerical variables sales, dcoilwtico, transactions, promotion after log1p',
            'Thêm feature is_earthquake': 'Add is_earthquake feature',
            'is_earthquake = 1 cho ngày 2016-04-16 và 1 tuần sau đó': 'is_earthquake = 1 for 2016-04-16 and the following week',
            'kiểm tra nhanh': 'quick check',
            'Phân tích xu hướng và mùa vụ:': 'Analyze trends and seasonality:',
            'Theo tuần': 'By week',
            'Theo tháng': 'By month',
            'Copy để tránh ghi đè': 'Copy to avoid overwriting',
            'Khôi phục sales gốc (vì cột \'sales\' đang là log1p)': 'Restore original sales (since the \'sales\' column is log1p)',
            'Tổng sales theo ngày (toàn hệ thống)': 'Total sales by day (entire system)',
            'Trung bình sales theo ngày (overall)': 'Average sales by day (overall)',
            'Lấy 20 ngày có tổng sales thấp nhất (chỉ giữ 2 cột)': 'Get 20 days with lowest total sales (keep only 2 columns)',
            'In kết quả': 'Print results',
            'Tạo cột is_Newyear: 1 nếu ngày là 1/1, ngược lại 0': 'Create is_Newyear column: 1 if date is 1/1, otherwise 0',
            'Kiểm tra nhanh': 'Quick check',
            'tự động lấy 200k điểm cuối rồi vẽ ACF/PACF': 'automatically take the last 200k points then plot ACF/PACF',
            'cấu hình nhanh': 'quick configuration',
            'số điểm tối đa để plot (an toàn cho máy)': 'maximum number of points to plot (safe for machine)',
            'số lag muốn xem': 'number of lags to view',
            'chuẩn bị dữ liệu': 'prepare data',
            'resample để giảm nhiễu / giảm số điểm': 'resample to reduce noise / reduce number of points',
            'với sales thường hợp lý là sum theo kỳ': 'for sales, it is usually reasonable to sum by period',
            'cắt còn MAX_N điểm cuối nếu quá dài': 'trim to last MAX_N points if too long',
            'vẽ ACF & PACF': 'plot ACF & PACF',
            'gợi ý MA(q)': 'suggests MA(q)',
            'PACF ổn định với method \'ywm\' hoặc \'ywadjusted\'': 'PACF is stable with method \'ywm\' or \'ywadjusted\'',
            'gợi ý AR(p)': 'suggests AR(p)',
            'Train test val split & Tạo Lag của sales': 'Train test val split & Create lags of sales',
            'Tạo lag features': 'Create lag features',
            'cột định danh series – đổi lại cho đúng tên cột của bạn': 'series identifier columns – change to match your column names',
            'ví dụ': 'example',
            'sort theo nhóm + thời gian để shift đúng': 'sort by group + time to shift correctly',
            'tạo lag theo từng series': 'create lags for each series',
            'thêm 18, 24 nếu muốn': 'add 18, 24 if you want',
            'bỏ hàng thiếu do lag (bị ở đầu mỗi nhóm)': 'remove missing rows due to lag (at the start of each group)',
            'Chuẩn bị': 'Prepare',
            'điều chỉnh cho đúng cột nhóm của bạn': 'adjust to match your group columns',
            'Tham số độ dài theo NGÀY (calendar day)': 'Length parameters by DAY (calendar day)',
            'Nếu muốn chỉ LẤY ĐÚNG 90 ngày cho train, đặt False.': 'If you want to take EXACTLY 90 days for train, set False.',
            'Mặc định True = lấy TẤT CẢ ngày trước mốc val_start vào train (train >= 90)': 'Default True = take ALL days before val_start for train (train >= 90)',
            'các mốc ngày duy nhất': 'unique date milestones',
            'Nếu chuỗi quá ngắn, fallback: ưu tiên test2, test1, val; phần còn lại là train': 'If series is too short, fallback: prioritize test2, test1, val; the rest is train',
            'cắt từ cuối chuỗi về trước theo tỉ lệ yêu cầu': 'cut from end of series backwards according to required ratio',
            'train = phần còn lại': 'train = the remainder',
            'Lấy đúng các cửa sổ từ CUỐI chuỗi': 'Take the correct windows from the END of series',
            'ngày cuối': 'last days',
            'ngày trước test2': 'days before test2',
            'ngày trước test1': 'days before test1',
            'gán nhãn theo date': 'assign labels by date',
            'Áp dụng theo từng chuỗi': 'Apply to each series',
            'Lấy ra các tập': 'Extract the sets',
            'Kiểm tra min/max date theo từng split (toàn cục)': 'Check min/max date for each split (global)',
            'chọn 2 chuỗi bất kỳ để kiểm tra': 'select any 2 series to check',
            'bạn có thể đổi sang family khác trong dataset của bạn': 'you can change to another family in your dataset',
            'Drop cột \'split\' khỏi các tập': 'Drop the \'split\' column from the sets',
            'Loại bỏ cột target trước khi phân loại features': 'Remove target column before classifying features',
            'tất cả cột trừ target': 'all columns except target',
            'categorical = object/string hoặc category dtype': 'categorical = object/string or category dtype',
            'hàm encode tuần hoàn': 'cyclical encoding function',
            'encode cho các biến tuần hoàn': 'encode cyclical variables',
            'drop các cột gốc (trừ year giữ nguyên)': 'drop original columns (except keep year)',
            'copy lại để tránh sửa nhầm': 'copy to avoid accidental modification',
            'khuyến nghị': 'recommended',
            'nếu có train_df, đồng bộ category theo train để tránh unseen codes': 'if train_df exists, synchronize categories with train to avoid unseen codes',
            'nếu muốn đồng bộ với train:': 'if you want to synchronize with train:',
            'hoặc truyền ref_df=train_df nếu có': 'or pass ref_df=train_df if available',
            'Cyclical encode cho các biến tuần hoàn ở val/test1/test2': 'Cyclical encode for cyclical variables in val/test1/test2',
            'nếu bạn dùng isocalendar().week thì có thể có 53; dayofyear có thể 366 (năm nhuận)': 'if you use isocalendar().week there can be 53; dayofyear can be 366 (leap year)',
            'dùng 53 an toàn cho ISO week': 'use 53 safely for ISO week',
            'an toàn cho năm nhuận': 'safe for leap year',
            'Drop cột gốc cyclical (giữ year)': 'Drop original cyclical columns (keep year)',
            'bỏ target ra khỏi tập features': 'remove target from feature set',
            
            # Additional words and phrases
            'ngày': 'day',
            'sự kiện': 'event',
            'các': 'the',
            'bảng': 'table',
            'theo': 'according to',
            'trong': 'in',
            'này': 'this',
            'được': 'is',
            'có': 'has',
            'thì': 'then',
            'là': 'is',
            'một': 'one',
            'số': 'number',
            'trên': 'on',
            'sau': 'after',
            'trước': 'before',
            'đầu': 'first',
            'cuối': 'last',
            'từ': 'from',
            'của': 'of',
            'và': 'and',
            'với': 'with',
            'cho': 'for',
            'để': 'to',
            'nhiều': 'many',
            'hơn': 'more than',
            'dữ liệu': 'data',
            'file': 'file',
            'cột': 'column',
            'hàng': 'row',
            'giá trị': 'value',
            'null': 'null',
            'train': 'train',
            'test': 'test',
            'val': 'val',
            'Tổng số ngày có nhiều hơn 1 event:': 'Total number of days with more than 1 event:',
            'Merge holiday đúng phạm vi:': 'Merge holidays with correct scope:',
            'Enable memory growth để tránh chiếm trọn VRAM': 'Enable memory growth to avoid occupying all VRAM',
            'Trung bình số lượng sản phẩm bán được theo ngày và cửa hàng cho từng nhóm hàng (family).': 'Average number of products sold by day and store for each product family.',
            'Loại cửa hàng': 'Store type',
            'Thêm thông tin cho ngày 2016-04-16: trận động đất 7.8 độ Richter ở Ecuador.': 'Add information for 2016-04-16: 7.8 Richter earthquake in Ecuador.',
            'Trong data, ngày này và một số ngày ngay sau đó có biến động sales và transactions bất thường': 'In the data, this day and some days immediately after have unusual sales and transaction fluctuations',
            '20 ngày có tổng sales thấp nhất:': '20 days with lowest total sales:',
            'Compile model với optimizer và loss function': 'Compile model with optimizer and loss function',
            'Mỗi series: nối tail(seq_len) của train + full val → cắt cửa sổ; chỉ trả đúng số điểm của val.': 'Each series: concatenate tail(seq_len) of train + full val → cut window; only return the correct number of val points.',
            'Tạo callbacks cho training': 'Create callbacks for training',
            'Vẽ biểu đồ training history': 'Plot training history',
            'Yield (x, y) theo đúng thứ tự thời gian, không rò rỉ val.': 'Yield (x, y) in correct time order, no val leakage.',
            'g_df đã sort theo DATE_COL và thuộc 1 series → yield (x_win, y_t, date_t).': 'g_df is sorted by DATE_COL and belongs to 1 series → yield (x_win, y_t, date_t).',
            'LayerNorm sau LSTM cuối:': 'LayerNorm after final LSTM:',
            'NEW: LayerNorm giữa các LSTM': 'NEW: LayerNorm between LSTMs',
            'DATASET BUILDER CHO MỖI CHẶNG': 'DATASET BUILDER FOR EACH STAGE',
            'Cấu hình': 'Configuration',
            'Fit scaler trên TRAIN': 'Fit scaler on TRAIN',
            'Sắp target theo (store, family, date) để đảm bảo thứ tự': 'Sort target by (store, family, date) to ensure order',
            'Dự đoán theo batch (streaming)': 'Predict by batch (streaming)',
            'Kiểm tra lại': 'Check again',
            'Lấy y_true trực tiếp từ target_df và inverse nếu cần': 'Get y_true directly from target_df and inverse if needed',
            'Tổng hợp nhanh theo series (xem series nào lệch nhất)': 'Quick summary by series (see which series is most off)',
            'Ghép bảng kết quả': 'Join result tables',
            'Bảo toàn bản gốc nếu cần': 'Preserve original if needed',
            'Ép dtype categorical cho cluster & store_nbr (val/test1/test2)': 'Cast dtype categorical for cluster & store_nbr (val/test1/test2)',
            'IN RA CÁC CỘT ĐÃ ENCODE XONG': 'PRINT OUT ENCODED COLUMNS',
            'Biểu đồ kiểm định lệch (calibration) và phân phối sai số': 'Calibration plot and error distribution',
        }
        
        # Common Vietnamese words to detect (for idempotency check)
        self.vietnamese_indicators = [
            'từ', 'của', 'và', 'với', 'cho', 'để', 'xử lý', 'ngày', 'sự kiện',
            'tuỳ chọn', 'các', 'bảng', 'theo', 'trong', 'này', 'được', 'có',
            'thì', 'là', 'một', 'số', 'trên', 'sau', 'trước', 'đầu', 'cuối'
        ]
    
    def contains_vietnamese(self, text):
        """Check if text contains Vietnamese characters or words."""
        # Check for Vietnamese-specific characters
        vietnamese_chars = re.compile(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', re.IGNORECASE)
        if vietnamese_chars.search(text):
            return True
        # Check for common Vietnamese words
        text_lower = text.lower()
        return any(word in text_lower for word in self.vietnamese_indicators)
    
    def translate_text(self, text):
        """Translate text using dictionary-based translation."""
        if not text or not text.strip():
            return text
        
        # Skip if doesn't contain Vietnamese
        if not self.contains_vietnamese(text):
            return text
        
        # Try exact match first (case-insensitive for keys, but preserve case in output)
        text_stripped = text.strip()
        for vn_text, en_text in self.translation_dict.items():
            if text_stripped.lower() == vn_text.lower():
                # Preserve original spacing/newlines
                leading_space = len(text) - len(text.lstrip())
                trailing_space = len(text) - len(text.rstrip())
                result = ' ' * leading_space + en_text + ' ' * trailing_space
                return result
        
        # Try partial replacements for phrases within the text
        result = text
        # Sort by length (longest first) to avoid replacing parts of longer phrases
        sorted_translations = sorted(self.translation_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for vn_text, en_text in sorted_translations:
            # Use word boundary matching where appropriate
            # For single Vietnamese words, use word boundaries
            if ' ' not in vn_text:
                # Single word - use word boundaries
                pattern = r'\b' + re.escape(vn_text) + r'\b'
                result = re.sub(pattern, en_text, result, flags=re.IGNORECASE)
            else:
                # Phrase - simple case-insensitive replacement
                # Find all occurrences with case preserved for surrounding text
                pattern = re.escape(vn_text)
                result = re.sub(pattern, en_text, result, flags=re.IGNORECASE)
        
        return result
    
    def extract_comment_from_line(self, line):
        """
        Extract the comment portion from a Python code line.
        Returns (code_part, comment_part) where comment_part includes the #.
        Ignores # inside string literals.
        """
        # Simple state machine to handle strings
        in_single_quote = False
        in_double_quote = False
        escaped = False
        
        for i, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            
            if char == '\\':
                escaped = True
                continue
            
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == '#' and not in_single_quote and not in_double_quote:
                # Found unquoted comment
                return line[:i], line[i:]
        
        # No comment found
        return line, ""
    
    def translate_code_cell(self, source_lines):
        """Translate comments and docstrings in code cell source."""
        translated_lines = []
        in_docstring = False
        docstring_delimiter = None
        
        for line in source_lines:
            translated_line = line
            
            # Check for docstring delimiters (""" or ''')
            triple_double_count = line.count('"""')
            triple_single_count = line.count("'''")
            
            # Handle multi-line docstrings
            if triple_double_count > 0 or triple_single_count > 0:
                # Determine which delimiter
                if triple_double_count > 0:
                    delimiter = '"""'
                    count = triple_double_count
                else:
                    delimiter = "'''"
                    count = triple_single_count
                
                # If we're starting or ending a docstring
                if count == 1:
                    if not in_docstring:
                        # Starting docstring
                        in_docstring = True
                        docstring_delimiter = delimiter
                        # Translate content after the opening delimiter
                        parts = line.split(delimiter, 1)
                        if len(parts) == 2:
                            translated_content = self.translate_text(parts[1])
                            translated_line = parts[0] + delimiter + translated_content
                    else:
                        # Ending docstring
                        # Translate content before the closing delimiter
                        parts = line.split(delimiter, 1)
                        if len(parts) == 2:
                            translated_content = self.translate_text(parts[0])
                            translated_line = translated_content + delimiter + parts[1]
                        in_docstring = False
                        docstring_delimiter = None
                elif count == 2:
                    # Single-line docstring
                    parts = line.split(delimiter)
                    if len(parts) >= 3:
                        # Translate the middle part
                        parts[1] = self.translate_text(parts[1])
                        translated_line = delimiter.join(parts)
            elif in_docstring:
                # We're inside a multi-line docstring - translate the entire line
                translated_line = self.translate_text(line)
            else:
                # Not in a docstring - handle regular comments
                code_part, comment_part = self.extract_comment_from_line(line)
                
                if comment_part and len(comment_part) > 1:  # Has a comment (more than just #)
                    # Extract the # and any whitespace
                    match = re.match(r'^(#\s*)', comment_part)
                    if match:
                        comment_prefix = match.group(1)
                        comment_text = comment_part[len(comment_prefix):]
                        
                        # Translate the comment text
                        translated_text = self.translate_text(comment_text)
                        translated_line = code_part + comment_prefix + translated_text
                
                # Also handle string literals in print statements
                if 'print(' in translated_line or 'print (' in translated_line:
                    # Translate content within string literals
                    def translate_string(match):
                        quote = match.group(1)
                        content = match.group(2)
                        translated_content = self.translate_text(content)
                        return f'{quote}{translated_content}{quote}'
                    
                    # Handle double-quoted strings
                    translated_line = re.sub(r'(")([^"]*?)"', translate_string, translated_line)
            
            translated_lines.append(translated_line)
        
        return translated_lines
    
    def split_markdown_preserve_code(self, text):
        """
        Split markdown text into segments, preserving fenced code blocks and inline code.
        Returns list of (is_code, content) tuples.
        """
        segments = []
        
        # Pattern for fenced code blocks
        fenced_pattern = r'(```|~~~).*?\1'
        # Pattern for inline code
        inline_pattern = r'`[^`]+`'
        
        # Combined pattern
        pattern = r'(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)'
        
        parts = re.split(pattern, text, flags=re.DOTALL)
        
        for part in parts:
            if not part:
                continue
            
            # Check if this is a code block
            if (part.startswith('```') and part.endswith('```')) or \
               (part.startswith('~~~') and part.endswith('~~~')) or \
               (part.startswith('`') and part.endswith('`') and '\n' not in part):
                segments.append((True, part))  # Code, don't translate
            else:
                segments.append((False, part))  # Text, translate
        
        return segments
    
    def translate_markdown_cell(self, source_lines):
        """Translate markdown cell, preserving code blocks and inline code."""
        # Join all lines into single text
        text = ''.join(source_lines)
        
        # Split into code and non-code segments
        segments = self.split_markdown_preserve_code(text)
        
        # Translate non-code segments
        translated_segments = []
        for is_code, content in segments:
            if is_code:
                translated_segments.append(content)
            else:
                translated_segments.append(self.translate_text(content))
        
        # Rejoin and split back into lines
        translated_text = ''.join(translated_segments)
        
        # Preserve original line structure as much as possible
        # This is tricky, so we'll just split on newlines and ensure we keep them
        if text.endswith('\n') and not translated_text.endswith('\n'):
            translated_text += '\n'
        
        # Convert back to list format matching original
        if len(source_lines) == 1:
            return [translated_text]
        else:
            # Try to preserve line breaks
            lines = translated_text.split('\n')
            # Ensure lines end with \n except possibly the last one
            result = []
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    result.append(line + '\n')
                elif line:  # Last line
                    result.append(line)
            return result if result else [translated_text]
    
    def translate_notebook(self, input_path, output_path=None):
        """Translate a Jupyter notebook file."""
        if output_path is None:
            output_path = input_path
        
        print(f"Loading notebook: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        total_cells = len(notebook['cells'])
        print(f"Processing {total_cells} cells...")
        
        for i, cell in enumerate(notebook['cells']):
            cell_type = cell['cell_type']
            print(f"  Cell {i+1}/{total_cells} ({cell_type})", end='... ')
            
            if cell_type == 'code':
                cell['source'] = self.translate_code_cell(cell['source'])
                print("done")
            elif cell_type == 'markdown':
                cell['source'] = self.translate_markdown_cell(cell['source'])
                print("done")
            else:
                print("skipped")
        
        print(f"Saving translated notebook: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        print("Translation complete!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python translate_ipynb_vi_to_en.py <notebook.ipynb> [output.ipynb]")
        print("  If output is not specified, the input file will be updated in place.")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    translator = NotebookTranslator()
    translator.translate_notebook(input_path, output_path)


if __name__ == '__main__':
    main()
