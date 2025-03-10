import datetime
import json
import traceback
import numpy as np
from supabase import create_client, Client
import torch
from app.core.config import settings
from app.libs.logger.log import log_error, log_info


class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url, settings.supabase_key)

    def get_image_metadata(self, image_id: str):
        response = self.client.table('image').select(
            '*').eq('id', image_id).execute()
        return response.data

    def update_image_labels(self, image_id: str, labels: dict):
        response = self.client.table('image').update(
            labels).eq('id', image_id).execute()
        return response.data

    def get_image_public_url(self, image_bucket_id: str, image_name: str):
        return self.client.storage.from_(image_bucket_id).get_public_url(image_name)

    def save_image_features_and_labels(self, image_bucket_id: str, image_name: str, labels: dict, image_features: torch.Tensor,  user_id: str = ''):
        if (user_id == '' or user_id is None):
            response = self.client.table('image').update({
                "updated_at": datetime.datetime.now().isoformat(),
                'labels': labels,
                'image_features': image_features,
            }).eq('image_bucket_id', image_bucket_id).eq('image_name', image_name).execute()
            return response.data[0]
        else:
            response = self.client.table('image').update({
                "updated_at": datetime.datetime.now().isoformat(),
                'labels': labels,
                'image_features': image_features,
                'uploader_id': user_id,
            }).eq('image_bucket_id', image_bucket_id).eq('image_name', image_name).execute()
            return response.data[0]

    def mark_image_done_face_detection(self, image_id: str):
        return self.client.table('image').update({
            "is_face_detection": True
        }).eq('id', image_id).execute()

    def update_person_table(self, face_encodings, face_locations, image_id, user_id, image_name):
        if len(face_locations) == 0 or len(face_encodings) == 0:
            log_info(f"No face found in image: {image_name}")
            return

        log_info(
            f"Found {len(face_locations)} faces in image: {image_name}")
        records = []
        for encoding, location in zip(face_encodings, face_locations):
            records.append({
                'embedding': np.array(encoding).tolist(),
                'coordinate': location,
                'image_id': image_id,
                'user_id': user_id,
            })
        try:
            self.client.table('person').insert(records).execute()
            return
        except Exception as e:
            log_error(
                f"Error update person table: {e}\n{traceback.format_exc()}")

    def get_all_user_person(self, user_id):
        try:
            response = self.client.table('person').select(
                '*, image(id, image_name, image_bucket_id, created_at, labels   )').eq('user_id', user_id).execute()
            return response.data
        except Exception as e:
            log_error(
                f"Error get all user person: {e}\n{traceback.format_exc()}")
            return []

    def insert_all_cluster_mapping(self, centroids):
        try:
            records = []
            for label, centroid in centroids.items():
                records.append({
                    'name': f'Person {label}',
                    'centroid': np.array(centroid).tolist()
                })

            response = self.client.table(
                'cluster_mapping').insert(records).execute()

            cluster_mapping = {}

            for i, record in enumerate(response.data):
                label_str = record['name'].split(' ')[1]
                cluster_mapping[label_str] = {
                    'id': record['id'],
                    'name': record['name'],
                }

            return cluster_mapping
        except Exception as e:
            log_error(
                f"Error insert all cluster mapping: {e}\n{traceback.format_exc()}")
            return []

    def update_person_cluster_id(self, person_ids, cluster_id):
        try:
            response = self.client.table('person').update({
                'cluster_id': cluster_id
            }).in_('id', person_ids).execute()
            return response.data
        except Exception as e:
            log_error(
                f"Error update person cluster id: {e}\n{traceback.format_exc()}")
            return []

    def create_and_update_cluster_for_noise_point(self, noise_points):
        request = []
        for person in noise_points:
            request.append({
                'name': f'Noise {person["id"]}',
                'centroid': person['embedding']
            })
        try:
            response = self.client.table(
                'cluster_mapping').insert(request).execute()
            cluster_mapping = {}

            for i, record in enumerate(response.data):
                label_str = record['name']
                cluster_mapping[label_str] = {
                    'cluster_id': record['id'],
                    'cluster_name': record['name'],
                    'person': [
                        {
                            'id': noise_points[i]['id'],
                            'coordinate': noise_points[i]['coordinate'],
                            'image_id': noise_points[i]['image']['id'],
                            'image_created_at': noise_points[i]['image']['created_at'],
                            'image_bucket_id': noise_points[i]['image']['image_bucket_id'],
                            'image_name': noise_points[i]['image']['image_name'],
                            'image_label': noise_points[i]['image']['labels']
                        }
                    ]
                }

            # update cluster_id for noise points
            for person in noise_points:
                cluster_id = cluster_mapping[f'Noise {person["id"]}']['cluster_id']
                self.client.table('person').update({
                    'cluster_id': cluster_id
                }).eq('id', person['id']).execute()

            return cluster_mapping
        except Exception as e:
            log_error(
                f"Error create and update cluster for noise point: {e}\n{traceback.format_exc()}")
            return []

    def get_all_cluster_mapping(self, user_id):
        try:
            response = self.client.table('person').select(
                '*, cluster_mapping(*)').eq('user_id', user_id).not_.is_('cluster_id', None).execute()
            return [{
                'id': person['cluster_mapping']['id'],
                'name': person['cluster_mapping']['name'],
                'centroid': json.loads(person['cluster_mapping']['centroid']),
            } for person in response.data]

        except Exception as e:
            log_error(
                f"Error get all cluster mapping: {e}\n{traceback.format_exc()}")
            return []

    def create_cluster(self, cluster_name, centroid):
        try:
            response = self.client.table('cluster_mapping').insert({
                'name': cluster_name,
                'centroid': centroid
            }).execute()
            return response.data[0]
        except Exception as e:
            log_error(
                f"Error create cluster: {e}\n{traceback.format_exc()}")
            raise e


def get_supabase_service():
    return SupabaseService()
