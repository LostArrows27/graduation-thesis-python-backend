from io import BytesIO
import json
import time

import numpy as np
import torch
from app.models.model import AIModel, FaceCategoryModel
from app.models.inference import AIInferenceService
from app.services.supabase_service import SupabaseService
import torch.nn.functional as F


class AIService:
    def __init__(self, supabase_service: SupabaseService):
        self.model = AIModel()
        self.face_model = FaceCategoryModel()
        self.inference_service = AIInferenceService(
            self.model, supabase_service, self.face_model)

    def save_text_search_history(self, text: str, user_id: str, threshold=0.24):

        text_features = self.model.get_text_features(text)

        search_history_id = self.inference_service.supabase_service.save_text_features_to_search_history(
            text, user_id, text_features.squeeze(0).tolist())

        # run supabase rpc
        # example:
        #         SELECT * FROM public.search_similar_images(
        #     '19b2701f-5a38-456b-ae63-5f5dd5225c2e'::UUID,
        #     0.24
        # );

        result = self.inference_service.supabase_service.query_image_by_search_history_id(
            search_history_id, user_id, threshold)

        return {
            'search_history_id': search_history_id,
            'result': result
        }

    def classify_image(self, image_bucket_id: str, image_name: str, image_id: str):
        return self.inference_service.classify_image(image_bucket_id, image_name, image_id)

    def update_image_labels(self, image_id: str, labels: dict):
        response_data = self.inference_service.supabase_service.update_image_labels(
            image_id, labels)
        return response_data

    def category_image_face(self, image_url: str):
        return self.inference_service.category_face(image_url)


def get_ai_service(supabase_service: SupabaseService):
    return AIService(supabase_service)


# from io import BytesIO
# import json
# import time

# import numpy as np
# import torch
# from app.models.model import AIModel, FaceCategoryModel
# from app.models.inference import AIInferenceService
# from app.services.supabase_service import SupabaseService
# import torch.nn.functional as F


# class AIService:
#     def __init__(self, supabase_service: SupabaseService):
#         self.model = AIModel()
#         # self.face_model = FaceCategoryModel()
#         # self.inference_service = AIInferenceService(
#         #     self.model, supabase_service, self.face_model)

#         text_features = self.model.get_text_features(
#             'a photo of traveling')

#         supabase_service.save_text_features_to_search_history(
#             'a photo of traveling', text_features.squeeze(0).tolist()
#         )

#         text_features = self.model.get_text_features(
#             'a photo of group of friends')

#         supabase_service.save_text_features_to_search_history(
#             'a photo of group of friends', text_features.squeeze(0).tolist()
#         )

#         # start = time.time()
#         # result = self.search_images_by_text_with_softmax_gpu(
#         #     supabase_service,
#         #     'food')

#         # print(f"Search time: {time.time() - start}")

#         # for item in result:
#         #     print(item.get('image_name'), item.get('similarity'))

#     def classify_image(self, image_bucket_id: str, image_name: str, image_id: str):
#         return self.inference_service.classify_image(image_bucket_id, image_name, image_id)

#     def update_image_labels(self, image_id: str, labels: dict):
#         response_data = self.inference_service.supabase_service.update_image_labels(
#             image_id, labels)
#         return response_data

#     def category_image_face(self, image_url: str):
#         return self.inference_service.category_face(image_url)

#     def search_images_by_text_with_softmax_gpu(self, supabase_service: SupabaseService, query_text: str, threshold=0.00):
#         """
#         Tìm kiếm ảnh dựa trên text sử dụng softmax và GPU

#         Args:
#             query_text: Chuỗi text tìm kiếm
#             threshold: Ngưỡng xác suất tối thiểu để chấp nhận kết quả

#         Returns:
#             list: Danh sách các ảnh có tương đồng cao với query_text
#         """
#         # 1. Lấy text features từ query và đưa lên GPU (nếu có)
#         text_features = self.model.get_text_features(query_text)

#         # Đảm bảo text_features ở trên thiết bị đúng
#         text_features = text_features.to(self.model.device)

#         # 2. Lấy tất cả image từ database
#         all_images = supabase_service.get_all_images()

#         # 3. Chuẩn bị dữ liệu
#         image_data = []
#         image_features_list = []

#         for image in all_images:
#             try:
#                 # Chuyển đổi image features từ nhiều định dạng có thể
#                 features = None
#                 if isinstance(image['image_features'], str):
#                     features = json.loads(image['image_features'])
#                 elif isinstance(image['image_features'], list):
#                     features = image['image_features']

#                 if features:
#                     image_data.append({
#                         'id': image['id'],
#                         'image_name': image.get('image_name', ''),
#                         'image_bucket_id': image.get('image_bucket_id', '')
#                     })
#                     image_features_list.append(features)
#             except (json.JSONDecodeError, KeyError, TypeError) as e:
#                 print(f"Error processing image: {e}")
#                 continue

#         if not image_features_list:
#             return []

#         # 4. Chuyển về PyTorch tensor và đưa lên GPU
#         image_features_tensor = torch.tensor(
#             image_features_list, dtype=torch.float32).to(self.model.device)

#         # 5. Chuẩn hóa vectors để đảm bảo dot product = cosine similarity
#         image_features_tensor = image_features_tensor / \
#             image_features_tensor.norm(dim=1, keepdim=True)
#         text_features = text_features / \
#             text_features.norm(dim=-1, keepdim=True)

#         # 6. Tính similarity với softmax trên GPU
#         with torch.no_grad():
#             # Nhân ma trận để tính similarity
#             similarities = 100.0 * (image_features_tensor @ text_features.T)

#             # Áp dụng softmax để chuyển thành xác suất
#             probs = F.softmax(similarities, dim=0)

#             # Chuyển về CPU cho xử lý tiếp theo
#             probs_cpu = probs.squeeze().cpu().numpy()

#         # 7. Lọc và tạo kết quả
#         results = []
#         for i, prob in enumerate(probs_cpu):
#             if prob > threshold:
#                 results.append({
#                     **image_data[i],
#                     'similarity': float(prob)
#                 })

#         # 8. Sắp xếp kết quả theo độ tương đồng giảm dần
#         results.sort(key=lambda x: x['similarity'], reverse=True)
#         return results


# def get_ai_service(supabase_service: SupabaseService):
#     return AIService(supabase_service)
