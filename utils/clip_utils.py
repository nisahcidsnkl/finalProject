import torch
import clip
from PIL import Image
import torchvision.transforms as transforms

class CLIPProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化CLIP处理器"""
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                               (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def encode_text(self, text):
        """编码文本"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image):
        """编码图像"""
        if isinstance(image, Image.Image):
            image = self.image_transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def compute_similarity(self, text_features, image_features):
        """计算文本和图像特征的相似度"""
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return similarity
    
    def get_text_embeddings(self, text_list):
        """获取多个文本的嵌入"""
        text_embeddings = []
        for text in text_list:
            embedding = self.encode_text(text)
            text_embeddings.append(embedding)
        return torch.cat(text_embeddings, dim=0)
    
    def get_image_embeddings(self, image_list):
        """获取多个图像的嵌入"""
        image_embeddings = []
        for image in image_list:
            embedding = self.encode_image(image)
            image_embeddings.append(embedding)
        return torch.cat(image_embeddings, dim=0)
    
    def find_best_match(self, text, image_list):
        """找到与文本最匹配的图像"""
        text_features = self.encode_text(text)
        image_features = self.get_image_embeddings(image_list)
        similarity = self.compute_similarity(text_features, image_features)
        best_match_idx = similarity.argmax().item()
        return best_match_idx, similarity[0][best_match_idx].item()

def get_clip_processor():
    """获取CLIP处理器实例"""
    return CLIPProcessor() 