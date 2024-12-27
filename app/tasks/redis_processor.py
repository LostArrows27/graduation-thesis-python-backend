import json
import logging
from app.services.ai_services import AIService
from app.services.redis_service import RedisService

logging.basicConfig(level=logging.INFO)


def process_label_job(ai_service: AIService, redis_service: RedisService, start_id='0'):
    while True:
        try:
            # If processing old stream, check for pending messages
            if start_id == '0':
                pending_info = redis_service.client.xpending(
                    'image_label_stream', 'image_label_group'
                )

                print(json.dumps(pending_info))
                if pending_info['pending'] == 0:
                    logging.info(
                        "No more old messages to process. Exiting old stream thread.")
                    break  # Exit the thread when no pending messages are found

            # Read messages from the stream
            messages = redis_service.read_from_stream(
                stream_name='image_label_stream',
                group_name='image_label_group',
                consumer_name='worker_1',
                count=1,
                block=0,
                start_id=start_id
            )

            # Process messages
            for stream, entries in messages:
                for entry_id, fields in entries:
                    image_id = fields['image_id']
                    image_bucket_id = fields['image_bucket_id']
                    image_name = fields['image_name']

                    try:
                        logging.info(f"Start processing {image_id}")

                        # Update Redis hash with job status
                        redis_service.update_hash(
                            f"image_job:{image_id}",
                            {
                                "image_bucket_id": image_bucket_id,
                                "image_name": image_name,
                                "label_status": "processing"
                            }
                        )

                        # Label the image
                        image_labels = ai_service.classify_image(
                            fields["image_bucket_id"],
                            fields["image_name"],
                            image_id
                        )

                        response_data = ai_service.update_image_labels(
                            image_id, image_labels
                        )

                        if response_data:
                            logging.info(
                                f"Labels for image {image_id} updated successfully")
                        else:
                            raise Exception(
                                f"Error updating labels for image {image_id}")

                        # Update Redis hash with labels
                        redis_service.update_hash(
                            f"image_job:{image_id}",
                            {
                                "labels": json.dumps(image_labels),
                                "label_status": "completed"
                            }
                        )

                        # Acknowledge the message
                        redis_service.ack_stream(
                            'image_label_stream', 'image_label_group', entry_id
                        )

                        # Optionally delete old stream entry
                        redis_service.delete_stream_entry(
                            'image_label_stream', entry_id
                        )

                        logging.info(f"End processing {entry_id}")

                    except Exception as e:
                        logging.error(
                            f"Error processing image {image_id}: {e}")

        except Exception as e:
            logging.error(f"Error reading from Redis stream: {e}")
