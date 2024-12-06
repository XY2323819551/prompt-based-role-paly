from typing import Optional, Dict
from dataclasses import dataclass
import os
import requests
import PyPDF2
from io import BytesIO
from tool_pool import BaseTool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PDFResult:
    """PDF解析结果的数据类"""
    success: bool
    message: str
    content: Optional[str] = None
    metadata: Optional[Dict] = None

class ArxivPDFTool(BaseTool):
    """用于下载和解析arXiv PDF文档的工具类"""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化ArxivPDF工具。

        Args:
            save_dir (str, optional): PDF文件保存目录，默认为当前目录下的'arxiv_pdfs'
        """
        self.save_dir = save_dir or os.path.join(os.getcwd(), 'arxiv_pdfs')
        os.makedirs(self.save_dir, exist_ok=True)
        super().__init__()

    def _download_pdf(self, url: str) -> BytesIO:
        """
        下载PDF文件。

        Args:
            url (str): PDF文件的URL

        Returns:
            BytesIO: 包含PDF内容的BytesIO对象
        """
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    def _extract_text(self, pdf_file: BytesIO) -> tuple[str, Dict]:
        """
        从PDF文件中提取文本和元数据。

        Args:
            pdf_file (BytesIO): PDF文件内容

        Returns:
            tuple[str, Dict]: 提取的文本内容和元数据
        """
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        metadata = reader.metadata
        return text, metadata

    def process_arxiv_pdf(self, pdf_url: str, save_local: bool = False) -> PDFResult:
        """
        处理arXiv PDF文档。

        Args:
            pdf_url (str): arXiv PDF文档的URL
            save_local (bool): 是否保存到本地，默认为False

        Returns:
            PDFResult: 包含处理结果的数据类实例
        """
        try:
            logger.info(f"开始下载PDF: {pdf_url}")
            pdf_file = self._download_pdf(pdf_url)
            
            logger.info("正在提取PDF内容...")
            content, metadata = self._extract_text(pdf_file)
            
            if save_local:
                filename = os.path.join(self.save_dir, f"{pdf_url.split('/')[-1]}")
                with open(filename, 'wb') as f:
                    pdf_file.seek(0)
                    f.write(pdf_file.read())
                logger.info(f"PDF已保存到: {filename}")
            
            return PDFResult(
                success=True,
                message="PDF处理成功",
                content=content,
                metadata=metadata
            )
            
        except requests.RequestException as e:
            return PDFResult(
                success=False,
                message=f"下载PDF时出错: {str(e)}"
            )
        except Exception as e:
            return PDFResult(
                success=False,
                message=f"处理PDF时出错: {str(e)}"
            )


def get_arxiv_pdf_content(pdf_url: str) -> str:
    """
    从arXiv PDF URL获取文档内容。arxiv_pdf工具获取PDF文档内容。
    
    此函数会下载指定的arXiv PDF文档并提取其文本内容。
    
    Args:
        pdf_url (str): arXiv PDF文档的完整URL。
                      例如: "https://arxiv.org/pdf/1706.03762.pdf"
    
    Returns:
        str: PDF文档的文本内容。如果处理失败，返回空字符串。
    
    Examples:
        >>> content = get_arxiv_pdf_content("https://arxiv.org/pdf/1706.03762.pdf")
        >>> print(len(content))
        42500
    
    Note:
        - URL应该是直接指向PDF文件的链接
        - 该函数依赖网络连接来下载PDF文件
        - 大型PDF文件的处理可能需要较长时间
    """
    tool = ArxivPDFTool()
    result = tool.process_arxiv_pdf(pdf_url, save_local=False)
    return result.content[:1000] if result.success else "内容提取失败"  # 先返回前1000个字符


if __name__ == "__main__":
    # 测试工具
    tool = ArxivPDFTool()
    
    # 测试用的arXiv PDF URL
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"  # Attention is All You Need论文
    
    print("测试PDF下载和解析:")
    result = tool.process_arxiv_pdf(test_url, save_local=True)
    
    if result.success:
        print("\n文档元数据:")
        print("\n文档总字符数: {len(result.content)}")
        print(result.metadata)
        print("\n文档内容预览 (前500个字符):")
        print(result.content[:500])
    else:
        print(f"处理失败: {result.message}") 