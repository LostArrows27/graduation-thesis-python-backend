import asyncio
import json
from typing import List
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import numpy as np
from sklearn.cluster import DBSCAN
from app.libs.logger.log import log_error, log_info
from app.services.redis_service import RedisService
from app.tasks.check_db_on_startup import cleanup_background_thread, start_background_processor
from app.tasks.db_listener import start_listener, stop_listener
from app.services.ai_services import AIService, get_ai_service
from app.services.supabase_service import SupabaseService
from app.tasks.redis_processor import start_stream_processors, stop_stream_processors
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import traceback
import os

from app.utils.compare_centroit import compare_centroids, remove_duplicates_by_image_name

os.environ['LOKY_MAX_CPU_COUNT'] = '10'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def reload_env():
    """Reload environment variables"""
    load_dotenv(override=True)


reload_env()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# dependency injection -> AI labeling service + Supabase service into -> bg_tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.supabase_service = SupabaseService()
    app.state.ai_service = AIService(app.state.supabase_service)
    app.state.redis_service = RedisService()

    # # # init consumer group
    app.state.redis_service.create_consumer_group(
        'image_label_stream', 'image_label_group')

    # # # start db not processed image processor
    # start_background_processor(
    #     app.state.ai_service,
    #     app.state.supabase_service, app.state.redis_service)
    # # start db change listener
    # start_listener(app.state.ai_service, app.state.supabase_service)
    # # start redis stream processors
    start_stream_processors(app.state.ai_service, app.state.redis_service)

    yield
    # stop_listener()
    stop_stream_processors()
    # cleanup_background_thread()

app = FastAPI(lifespan=lifespan)


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


def get_redis_service(request: Request) -> RedisService:
    return request.app.state.redis_service


def get_supabase_service(request: Request) -> SupabaseService:
    return request.app.state.supabase_service


class PersonClustering(BaseModel):
    user_id: str


# return person group + noise point group
# each group contain cluster_id, cluster_name, person[]
@app.post("/api/person-clustering")
def person_clustering(request: PersonClustering, supabase_service: SupabaseService = Depends(get_supabase_service)):
    try:
        user_id = request.user_id
        # get all person of the user
        person_list = supabase_service.get_all_user_person(user_id)

        if (person_list is None) or (len(person_list) == 0):
            return {"status": "success", "data": {}}

        eps = 0.41  # or 0.4-4
        min_samples = 4  # or 0.41-3

        log_info('user request clustering person')

        embeddings = [json.loads(person['embedding'])
                      for person in person_list]

        dbscan = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples)
        labels = dbscan.fit(embeddings)

        # group person by labels / noise -> -1
        person_groups = {}
        noise_points = []
        for i, label in enumerate(labels.labels_):
            if label == -1:
                noise_points.append(person_list[i])
            else:
                label_str = str(label)
                if label_str not in person_groups:
                    person_groups[label_str] = []
                person_groups[label_str].append(person_list[i])

        is_had_old_cluster = False
        for person in person_list:
            if person['cluster_id'] is not None:
                is_had_old_cluster = True
                break

        # calculate centroid for each new cluster
        centroids = {}
        for label, group in person_groups.items():
            cluster_embeddings = [json.loads(
                person['embedding']) for person in group]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids[label] = centroid

        # case 1 -> no cluster_id
        if not is_had_old_cluster:
            log_info("No cluster_id")
            # 1. insert all cluster to db
            cluster_ids = supabase_service.insert_all_cluster_mapping(
                centroids)

            # 2. update all person in each cluster with cluster_id
            # cluster_id -> {label: id}
            for label, group in person_groups.items():
                cluster_id = cluster_ids[str(label)]['id']
                cluster_name = cluster_ids[str(label)]['name']

                person_return_data = []
                for person in group:
                    person_return_data.append({
                        'id': person['id'],
                        'coordinate': person['coordinate'],
                        'image_id': person['image']['id'],
                        'image_created_at': person['image']['created_at'],
                        'image_bucket_id': person['image']['image_bucket_id'],
                        'image_name': person['image']['image_name'],
                        'image_label': person['image']['labels'],
                    })

                person_groups[label] = {
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'person': person_return_data
                }

                update_person_id = [person['id'] for person in group]

                supabase_service.update_person_cluster_id(
                    update_person_id, cluster_id)

            # 3. create new cluster_id for noise points
            noise_point_group = supabase_service.create_and_update_cluster_for_noise_point(
                noise_points)

            # only take group with >= 2 person
            person_groups = {
                k: v for k, v in person_groups.items() if len(v['person']) >= 2}
            noise_point_group = {
                k: v for k, v in noise_point_group.items() if len(v['person']) >= 2}

            # in each group, remove person with same image_url
            for label, group in person_groups.items():
                group['person'] = remove_duplicates_by_image_name(
                    group['person'])

            for label, group in noise_point_group.items():
                group['person'] = remove_duplicates_by_image_name(
                    group['person'])

            # Filter groups again to ensure they still have >= 2 persons after deduplication
            person_groups = {
                k: v for k, v in person_groups.items() if len(v['person']) >= 2}
            noise_point_group = {
                k: v for k, v in noise_point_group.items() if len(v['person']) >= 2}

            combined_group = {**person_groups, **noise_point_group}
            return {"status": "success", "data": combined_group}
            # return {"status": "success", "data": {"person_groups": person_groups, "noise_points": noise_point_group}}

        # case 2 -> has cluster_id
        # 1. calculate threshold between old and new cluster
        # 2. update person with new cluster_id if threshold < ....
        # 3. insert all new cluster to db
        # 4. update person with new cluster_id
        else:
            # 1. get all old cluster
            old_clusters = supabase_service.get_all_cluster_mapping(
                user_id=user_id)

            new_clusters = centroids

            combine_cluster_group = compare_centroids(new_clusters, old_clusters,
                                                      person_groups, noise_points, supabase_service)

            # return {"status": "success", "data": {"person_groups": new_cluster_group, "noise_points": old_cluster_group}}
            return {"status": "success", "data": combine_cluster_group}

    except Exception as e:
        log_error(f"Error person clutering API: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": "Error in person clustering."}


class ImageRequest(BaseModel):
    image_bucket_id: str
    image_name: str
    image_id: str
    user_id: str


class ImageList(BaseModel):
    image_bucket_id: str
    image_name: str
    image_id: str


class ImageBatchRequest(BaseModel):
    user_id: str
    data: List[ImageList]


@app.post("/api/classify-images")
async def classify_images(request: ImageBatchRequest, service: AIService = Depends(get_ai_service), redis_service: RedisService = Depends(get_redis_service)):
    # if len(request.data) > 3:
    #     return {"status": "error", "message": "Only up to 3 images are allowed."}

    # check user id
    if request.user_id == '' or request.user_id is None:
        return {"status": "error", "message": "User id is required."}

    async def process_image(image_request: ImageRequest):
        try:
            image_id = image_request.image_id
            image_bucket_id = image_request.image_bucket_id
            image_name = image_request.image_name

            # update redis label job -> processing
            redis_service.update_image_label_job(
                image_id, image_bucket_id, image_name
            )

            image_url = service.inference_service.supabase_service.get_image_public_url(
                image_bucket_id, image_name)

            results, image_features = service.classify_image(
                image_bucket_id, image_name, image_url)

            # update redis label job -> completed
            redis_service.update_hash(
                f"image_job:{image_id}",
                {
                    "labels": json.dumps(results),
                    "label_status": "completed"
                }
            )

            supabase_service: SupabaseService = service.inference_service.supabase_service
            image_row = supabase_service.save_image_features_and_labels(
                image_bucket_id, image_name, results, image_features.squeeze(0).tolist(), user_id=request.user_id)
            image_row.pop('image_features')
            return image_row
        except Exception as e:
            log_error(e)
            log_error(f"Error at API endpoint: {e}\n{traceback.format_exc()}")
            #  throw error to the caller
            raise Exception(e)

    try:
        tasks = [process_image(img_req) for img_req in request.data]
        image_rows = await asyncio.gather(*tasks)
    except Exception as e:
        # add statuscode 500
        return {"status": "error", "message": str(e)}
    return {"status": "success", "data": image_rows}


class QueryImageRequest(BaseModel):
    user_id: str
    query: str
    threshold: float


@app.post("/api/query-image")
def query_image(request: QueryImageRequest, service: AIService = Depends(get_ai_service)):
    # check user id
    if request.user_id == '' or request.user_id is None:
        return {"status": "error", "message": "User id is required."}

    if request.threshold < 0 or request.threshold > 1:
        return {"status": "error", "message": "Threshold must be between 0 and 1."}

    if request.query == '' or request.query is None:
        return {"status": "error", "message": "Query is required."}

    try:
        result = service.save_text_search_history(
            request.query, request.user_id, request.threshold)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# hello world test endpoint


@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.post("/api/classify-image")
# def classify_image(request: ImageRequest, service: AIService = Depends(get_ai_service), redis_service: RedisService = Depends(get_redis_service)):
#     # check user id
#     if request.user_id == '' or request.user_id is None:
#         return {"status": "error", "message": "User id is required."}

#     try:
#         image_id = request.image_id
#         image_bucket_id = request.image_bucket_id
#         image_name = request.image_name
#         image_url = service.inference_service.supabase_service.get_image_public_url(
#             image_bucket_id, image_name)

#         # update redis label job -> processing
#         redis_service.update_image_label_job(
#             image_id, image_bucket_id, image_name
#         )

#         results, image_features = service.classify_image(
#             image_bucket_id, image_name, image_url)

#         # update redis label job -> completed
#         redis_service.update_hash(
#             f"image_job:{image_id}",
#             {
#                 "labels": json.dumps(results),
#                 "label_status": "completed"
#             }
#         )

#         supabase_service: SupabaseService = service.inference_service.supabase_service
#         image_row = supabase_service.save_image_features_and_labels(
#             image_bucket_id, image_name, results, image_features.squeeze(0).tolist(), user_id=request.user_id)

#         # remove image_features from response
#         image_row.pop('image_features')

#         return {"status": "success", "data": image_row}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
