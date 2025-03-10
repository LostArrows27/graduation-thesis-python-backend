import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.libs.logger.log import log_info
from app.services.supabase_service import SupabaseService

# for each new cluster, compare with all old clusters
# -> find the most similar old cluster
# 1. new_cluster -> {[new_cluster_id_label]: [new_cluster_centroid]}
# 2. old_clusters -> [{id, name, centroid}]
# 3. noise_centroids -> {[Noise {person["id"]}]: [person['embedding']]}
# 4. person_groups -> {[new_cluster_id_label]: [person]}
# 5. noise_points -> [person]

# RETURN:
# 1. person_groups -> {[label]: {person: [id+url+coordinate], cluster_id, cluster_name}}}
# 2. noise_groups -> {[label]: {person: [id+url+coordinate], cluster_id, cluster_name}}}


def compare_centroids(new_clusters, old_clusters, person_groups, noise_points, supabase_service: SupabaseService):
    # Create centroids for noise points
    noise_centroids = {
        f'Noise {person["id"]}': np.array(json.loads(person['embedding'])) for person in noise_points
    }

    new_cluster_group = {}
    old_cluster_group = {}

    # Loop through each new cluster and compare with all old clusters
    for label, new_centroid in new_clusters.items():
        cluster_group = create_or_update_cluster(label, np.array(
            new_centroid), old_clusters, person_groups[label], supabase_service)
        new_cluster_group.update(cluster_group)

    # Loop through each noise cluster and compare with all old clusters
    for label, new_noise_centroid in noise_centroids.items():
        # find person in noise_point by label
        person = [person for person in noise_points if person['id']
                  == int(label.split(' ')[1])][0]
        cluster_group = create_or_update_cluster(
            label, new_noise_centroid, old_clusters, [person], supabase_service, is_noise=True)
        old_cluster_group.update(cluster_group)

    # only take group with >= 2 person
    new_cluster_group = {
        k: v for k, v in new_cluster_group.items() if len(v['person']) >= 2}
    old_cluster_group = {
        k: v for k, v in old_cluster_group.items() if len(v['person']) >= 2}

    # in each group, remove person with same image_url
    for label, group in new_cluster_group.items():
        group['person'] = remove_duplicates_by_image_name(group['person'])

    for label, group in old_cluster_group.items():
        group['person'] = remove_duplicates_by_image_name(group['person'])

    # Filter groups again to ensure they still have >= 2 persons after deduplication
    new_cluster_group = {
        k: v for k, v in new_cluster_group.items() if len(v['person']) >= 2}
    old_cluster_group = {
        k: v for k, v in old_cluster_group.items() if len(v['person']) >= 2}

    # combine new_cluster_group and old_cluster_group
    results_group = {**new_cluster_group, **old_cluster_group}

    return results_group


def create_or_update_cluster(label, new_centroid, clusters, person_group, supabase_service: SupabaseService, is_noise=False):
    threshold = 0.95
    old_centroids = [np.array(cluster['centroid']) for cluster in clusters]
    similarities = cosine_similarity([new_centroid], old_centroids)
    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[0, best_match_index]

    results_group = {}

    if best_match_similarity >= threshold:
        # Pop the matched old cluster from the list
        old_clusters = clusters.pop(best_match_index)

        # Update the person group with old_cluster_id
        old_cluster_id = old_clusters['id']
        person_ids = [person['id'] for person in person_group]
        supabase_service.update_person_cluster_id(person_ids, old_cluster_id)

        # put in results_group
        person_group_results = []

        for person in person_group:
            person_group_results.append({
                'id': person['id'],
                'image_id': person['image']['id'],
                'image_created_at': person['image']['created_at'],
                'image_bucket_id': person['image']['image_bucket_id'],
                'image_name': person['image']['image_name'],
                'image_label': person['image']['labels'],
                'coordinate': person['coordinate']})

        results_group[label] = {
            'person': person_group_results,
            'cluster_id': old_cluster_id,
            'cluster_name': old_clusters['name']
        }

    else:
        # Create new cluster
        label_id = ""
        if (label.startswith('Noise') or label.startswith('Person')):
            label_id = label.split(' ')[1]
        cluster_name = f"{'Noise ' if is_noise else 'Person '}{label_id}"
        centroid = new_centroid.tolist()
        new_cluster = supabase_service.create_cluster(cluster_name, centroid)

        # Update person cluster_id
        new_cluster_id = new_cluster['id']
        person_ids = [person['id'] for person in person_group]
        supabase_service.update_person_cluster_id(person_ids, new_cluster_id)

        person_group_results = []

        # put in results_group
        for person in person_group:
            person_group_results.append({
                'id': person['id'],
                'image_id': person['image']['id'],
                'image_created_at': person['image']['created_at'],
                'image_bucket_id': person['image']['image_bucket_id'],
                'image_name': person['image']['image_name'],
                'image_label': person['image']['labels'],
                'coordinate': person['coordinate']})

        results_group[label] = {
            'person': person_group_results,
            'cluster_id': new_cluster_id,
            'cluster_name': cluster_name
        }

    return results_group


def remove_duplicates_by_image_name(person_list):
    seen_image_name = set()
    unique_persons = []

    for person in person_list:
        image_name = person['image_name']
        if image_name not in seen_image_name:
            seen_image_name.add(image_name)
            unique_persons.append(person)

    return unique_persons

# # Example usage:
# new_clusters = {
#     1: [0.1, 0.2, 0.3, 0.4],  # example centroid of new cluster 1
#     2: [0.5, 0.6, 0.7, 0.8],  # example centroid of new cluster 2
# }

# old_clusters = [
#     {'id': 1, 'name': 'Cluster A', 'centroid': [0.1, 0.2, 0.3, 0.4]},  # example old cluster A
#     {'id': 2, 'name': 'Cluster B', 'centroid': [0.9, 1.0, 1.1, 1.2]},  # example old cluster B
#     {'id': 3, 'name': 'Cluster C', 'centroid': [0.2, 0.3, 0.4, 0.5]},  # example old cluster C
# ]
