import fitz
import string


def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    return text


def extract_main_text_from_pdf(pdf_path):

    ascii_chars = set(string.printable)

    with fitz.open(pdf_path) as pdf:
        text = ""
        for i, page in enumerate(pdf):
            if i < len(pdf) - 2:
                blocks = page.get_text("blocks")
                main_text_blocks = [block for block in blocks if not block[4].strip(
                ).startswith(("Figure", "Table"))]
                page_text = "\n".join(block[4] for block in main_text_blocks)
                page_text = ''.join(
                    filter(lambda x: x in ascii_chars, page_text))
                text += page_text + "\n"
    return text


def chunk_string(s, chunk_size):
    return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]



def url2fn(url: str):
    fn = url
    try:
        if 'https' in fn:
            fn = fn.replace("https://", "")
        elif 'http' in fn:
            fn = fn.replace("http://", "")
        return fn.replace("/", "-").replace(":", "_")
    except Exception as e:
        raise ValueError(f'Incorrect file name {e}')

