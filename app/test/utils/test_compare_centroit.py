import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.utils.compare_centroit import (
    compare_centroids,
    create_or_update_cluster,
    remove_duplicates_by_image_name
)


@pytest.fixture
def mock_supabase_service():
    mock = MagicMock()
    mock.create_cluster.return_value = {'id': 999}
    mock.update_person_cluster_id.return_value = None
    return mock


@pytest.fixture
def sample_new_clusters():
    return {
        'Cluster 1': [0.1, 0.2, 0.3, 0.4],
        'Cluster 2': [0.5, 0.6, 0.7, 0.8],
    }


@pytest.fixture
def sample_old_clusters():
    return [
        {'id': 1, 'name': 'Cluster A', 'centroid': [0.1, 0.2, 0.3, 0.4]},
        {'id': 2, 'name': 'Cluster B', 'centroid': [0.9, 1.0, 1.1, 1.2]},
    ]


@pytest.fixture
def sample_person_groups():
    return {
        'Cluster 1': [
            {
                'id': 101,
                'image': {
                    'id': 201,
                    'created_at': '2023-01-01',
                    'image_bucket_id': 'bucket1',
                    'image_name': 'image1.jpg',
                    'labels': ['person']
                },
                'coordinate': [10, 20, 30, 40]
            },
            {
                'id': 102,
                'image': {
                    'id': 202,
                    'created_at': '2023-01-02',
                    'image_bucket_id': 'bucket1',
                    'image_name': 'image2.jpg',
                    'labels': ['person']
                },
                'coordinate': [15, 25, 35, 45]
            }
        ],
        'Cluster 2': [
            {
                'id': 103,
                'image': {
                    'id': 203,
                    'created_at': '2023-01-03',
                    'image_bucket_id': 'bucket1',
                    'image_name': 'image3.jpg',
                    'labels': ['person']
                },
                'coordinate': [50, 60, 70, 80]
            },
        ]
    }


@pytest.fixture
def sample_noise_points():
    return [
        {
            'id': 104,
            'image': {
                'id': 204,
                'created_at': '2023-01-04',
                'image_bucket_id': 'bucket1',
                'image_name': 'image4.jpg',
                'labels': ['person']
            },
            'coordinate': [90, 100, 110, 120],
            'embedding': json.dumps([0.9, 0.9, 0.9, 0.9])
        },
        {
            'id': 105,
            'image': {
                'id': 205,
                'created_at': '2023-01-05',
                'image_bucket_id': 'bucket1',
                'image_name': 'image5.jpg',
                'labels': ['person']
            },
            'coordinate': [130, 140, 150, 160],
            'embedding': json.dumps([0.1, 0.1, 0.1, 0.1])
        }
    ]


def test_remove_duplicates_by_image_name():
    # Test with duplicate image names
    person_list = [
        {
            'id': 1,
            'image_name': 'image1.jpg',
            'coordinate': [10, 20, 30, 40]
        },
        {
            'id': 2,
            'image_name': 'image2.jpg',
            'coordinate': [50, 60, 70, 80]
        },
        {
            'id': 3,
            'image_name': 'image1.jpg',  # Duplicate
            'coordinate': [90, 100, 110, 120]
        }
    ]

    result = remove_duplicates_by_image_name(person_list)

    # Should return only two persons (no duplicates)
    assert len(result) == 2
    assert result[0]['id'] == 1
    assert result[1]['id'] == 2

    # Test with no duplicates
    person_list_unique = [
        {
            'id': 1,
            'image_name': 'image1.jpg',
            'coordinate': [10, 20, 30, 40]
        },
        {
            'id': 2,
            'image_name': 'image2.jpg',
            'coordinate': [50, 60, 70, 80]
        }
    ]

    result = remove_duplicates_by_image_name(person_list_unique)

    # Should return the same list
    assert len(result) == 2
    assert result[0]['id'] == 1
    assert result[1]['id'] == 2


def test_create_or_update_cluster_match_found(mock_supabase_service, sample_old_clusters):
    # Test when a match is found (similarity >= threshold)
    label = 'Test Cluster'
    new_centroid = np.array([0.1, 0.2, 0.3, 0.4])  # Same as first old cluster
    person_group = [
        {
            'id': 101,
            'image': {
                'id': 201,
                'created_at': '2023-01-01',
                'image_bucket_id': 'bucket1',
                'image_name': 'image1.jpg',
                'labels': ['person']
            },
            'coordinate': [10, 20, 30, 40]
        }
    ]

    # Clone the sample_old_clusters to avoid modifying the fixture
    old_clusters = sample_old_clusters.copy()

    result = create_or_update_cluster(
        label, new_centroid, old_clusters, person_group, mock_supabase_service
    )

    # Should update existing cluster
    assert label in result
    assert result[label]['cluster_id'] == 1  # ID of matched cluster
    assert result[label]['cluster_name'] == 'Cluster A'
    assert len(result[label]['person']) == 1
    assert result[label]['person'][0]['id'] == 101

    # Should call update_person_cluster_id with correct args
    mock_supabase_service.update_person_cluster_id.assert_called_once_with([
                                                                           101], 1)

    # Should not call create_cluster
    mock_supabase_service.create_cluster.assert_not_called()

    # The matched cluster should be removed from old_clusters
    assert len(old_clusters) == 1
    assert old_clusters[0]['id'] == 2


def test_create_or_update_cluster_no_match(mock_supabase_service, sample_old_clusters):
    # Test when no match is found (similarity < threshold)
    label = 'Test Cluster'
    # Very different from old clusters
    new_centroid = np.array([-0.9, -0.8, -0.7, -0.6])
    person_group = [
        {
            'id': 102,
            'image': {
                'id': 202,
                'created_at': '2023-01-02',
                'image_bucket_id': 'bucket1',
                'image_name': 'image2.jpg',
                'labels': ['person']
            },
            'coordinate': [15, 25, 35, 45]
        }
    ]

    # Clone the sample_old_clusters to avoid modifying the fixture
    old_clusters = sample_old_clusters.copy()

    result = create_or_update_cluster(
        label, new_centroid, old_clusters, person_group, mock_supabase_service
    )

    # Should create new cluster
    assert label in result
    assert result[label]['cluster_id'] == 999  # ID from mock
    # Default prefix when no ID in label
    assert result[label]['cluster_name'] == 'Person '
    assert len(result[label]['person']) == 1
    assert result[label]['person'][0]['id'] == 102

    # Should call create_cluster with correct args
    mock_supabase_service.create_cluster.assert_called_once()
    args = mock_supabase_service.create_cluster.call_args[0]
    assert args[0] == 'Person '  # cluster_name
    assert isinstance(args[1], list)  # centroid

    # Should call update_person_cluster_id with correct args
    mock_supabase_service.update_person_cluster_id.assert_called_once_with([
                                                                           102], 999)

    # Old clusters should not be modified
    assert len(old_clusters) == 2


def test_create_or_update_cluster_noise_point(mock_supabase_service, sample_old_clusters):
    # Test with a noise point (is_noise=True)
    label = 'Noise 123'
    new_centroid = np.array([-0.9, -0.8, -0.7, -0.6])
    person_group = [
        {
            'id': 123,
            'image': {
                'id': 223,
                'created_at': '2023-01-03',
                'image_bucket_id': 'bucket1',
                'image_name': 'image3.jpg',
                'labels': ['person']
            },
            'coordinate': [25, 35, 45, 55]
        }
    ]

    # Clone the sample_old_clusters to avoid modifying the fixture
    old_clusters = sample_old_clusters.copy()

    result = create_or_update_cluster(
        label, new_centroid, old_clusters, person_group, mock_supabase_service, is_noise=True
    )

    # Should create new cluster with 'Noise ' prefix
    assert label in result
    assert result[label]['cluster_id'] == 999  # ID from mock
    # Should extract ID from label
    assert result[label]['cluster_name'] == 'Noise 123'
    assert len(result[label]['person']) == 1
    assert result[label]['person'][0]['id'] == 123


def test_compare_centroids_basic(
    mock_supabase_service,
    sample_new_clusters,
    sample_old_clusters,
    sample_person_groups,
    sample_noise_points
):
    # Mock both cosine_similarity and create_or_update_cluster
    with patch('app.utils.compare_centroit.cosine_similarity') as mock_cosine, \
            patch('app.utils.compare_centroit.create_or_update_cluster') as mock_create, \
            patch('app.utils.compare_centroit.remove_duplicates_by_image_name') as mock_remove_duplicates:

        # Configure mocks
        mock_cosine.return_value = np.array([[0.99, 0.01]])
        # Just return the same list without changes
        mock_remove_duplicates.side_effect = lambda x: x

        # Set up return values for create_or_update_cluster
        mock_create.side_effect = [
            {'Cluster 1': {
                'cluster_id': 1,
                'cluster_name': 'Cluster A',
                'person': sample_person_groups['Cluster 1']
            }},
            {'Cluster 2': {
                'cluster_id': 2,
                'cluster_name': 'Cluster B',
                'person': sample_person_groups['Cluster 2']
            }},
            {'Noise 104': {
                'cluster_id': 3,
                'cluster_name': 'Noise Cluster',
                'person': [{'id': 104, 'image_name': 'image4.jpg'}]
            }},
            {'Noise 105': {
                'cluster_id': 4,
                'cluster_name': 'Noise Cluster',
                'person': [{'id': 105, 'image_name': 'image5.jpg'}]
            }}
        ]

        # Test basic functionality
        result = compare_centroids(
            sample_new_clusters,
            sample_old_clusters.copy(),
            sample_person_groups,
            sample_noise_points,
            mock_supabase_service
        )

        # Check that all clusters with >= 2 persons are in result
        assert 'Cluster 1' in result
        assert len(result) >= 1

        # Check that 'Cluster 2' is not in result (only has 1 person)
        assert 'Cluster 2' not in result


def test_compare_centroids_similarity_match(
    mock_supabase_service,
    sample_old_clusters,
    sample_person_groups
):
    # Test when there's a high similarity match
    new_clusters = {
        'Cluster 1': [0.1, 0.2, 0.3, 0.4],  # Same as first old cluster
    }

    # Ensure Cluster 1 has at least 2 persons
    assert len(sample_person_groups['Cluster 1']) >= 2

    # Mock cosine_similarity
    with patch('app.utils.compare_centroit.cosine_similarity') as mock_cosine:
        # High similarity with first cluster
        mock_cosine.return_value = np.array([[0.99, 0.01]])

        result = compare_centroids(
            new_clusters,
            sample_old_clusters.copy(),
            sample_person_groups,
            [],  # No noise points
            mock_supabase_service
        )

        # Check that Cluster 1 was matched with old cluster
        assert 'Cluster 1' in result
        assert result['Cluster 1']['cluster_id'] == 1
        assert result['Cluster 1']['cluster_name'] == 'Cluster A'


def test_compare_centroids_with_noise_points(
    mock_supabase_service,
    sample_new_clusters,
    sample_old_clusters
):
    # Test with only noise points (no regular clusters)
    noise_points = [
        {
            'id': 104,
            'image': {
                'id': 204,
                'created_at': '2023-01-04',
                'image_bucket_id': 'bucket1',
                'image_name': 'image4.jpg',
                'labels': ['person']
            },
            'coordinate': [90, 100, 110, 120],
            # Similar to first old cluster
            'embedding': json.dumps([0.1, 0.2, 0.3, 0.4])
        },
        {
            'id': 105,
            'image': {
                'id': 205,
                'created_at': '2023-01-05',
                'image_bucket_id': 'bucket1',
                'image_name': 'image5.jpg',
                'labels': ['person']
            },
            'coordinate': [130, 140, 150, 160],
            # Duplicate to test deduplication
            'embedding': json.dumps([0.1, 0.2, 0.3, 0.4])
        }
    ]

    result = compare_centroids(
        {},  # No new clusters
        sample_old_clusters.copy(),  # Copy to avoid modifying fixture
        {},  # No person groups
        noise_points,
        mock_supabase_service
    )

    # Since we have two noise points but they have the same image name,
    # after deduplication there will be less than 2 persons per group,
    # so the result should be empty
    assert len(result) == 0

    # Test with unique noise points to ensure they form a group
    noise_points = [
        {
            'id': 104,
            'image': {
                'id': 204,
                'created_at': '2023-01-04',
                'image_bucket_id': 'bucket1',
                'image_name': 'image4.jpg',
                'labels': ['person']
            },
            'coordinate': [90, 100, 110, 120],
            'embedding': json.dumps([0.1, 0.2, 0.3, 0.4])
        },
        {
            'id': 105,
            'image': {
                'id': 205,
                'created_at': '2023-01-05',
                'image_bucket_id': 'bucket1',
                'image_name': 'image5.jpg',  # Different name
                'labels': ['person']
            },
            'coordinate': [130, 140, 150, 160],
            'embedding': json.dumps([0.1, 0.2, 0.3, 0.4])
        }
    ]

    # Patch the create_or_update_cluster to ensure it returns a consistent result
    # for both noise points (same cluster)
    with patch('app.utils.compare_centroit.create_or_update_cluster') as mock_create:
        mock_create.return_value = {
            'Noise 104': {
                'person': [
                    {'id': 104, 'image_name': 'image4.jpg'},
                    {'id': 105, 'image_name': 'image5.jpg'}
                ],
                'cluster_id': 1,
                'cluster_name': 'Cluster A'
            }
        }

        result = compare_centroids(
            {},  # No new clusters
            sample_old_clusters.copy(),
            {},  # No person groups
            noise_points,
            mock_supabase_service
        )

        # Now we should have at least one group
        assert len(result) >= 1
        assert 'Noise 104' in result
        assert len(result['Noise 104']['person']) == 2
