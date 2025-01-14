import json
import logging
from concurrent.futures import ThreadPoolExecutor
from app.services.ai_services import AIService
from app.services.redis_service import RedisService
import threading

logging.basicConfig(level=logging.INFO)

old_stream_thread = None
new_stream_thread = None
stop_event = threading.Event()  # Global stop event


def process_message(ai_service: AIService, redis_service: RedisService, entry_id, fields):
    image_id = fields['image_id']
    image_bucket_id = fields['image_bucket_id']
    image_name = fields['image_name']

    print(f"Processing image {image_id}")

    try:
        logging.info(f"Start processing {image_id}")

        # Update Redis hash with job status (max 3 hours)
        redis_service.update_hash(
            f"image_job:{image_id}",
            {
                "image_bucket_id": image_bucket_id,
                "image_name": image_name,
                "label_status": "processing"
            }
        )

        redis_service.set_ttl(f"image_job:{image_id}", 10800)

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
            logging.info(f"Labels for image {image_id} updated successfully")
        else:
            raise Exception(f"Error updating labels for image {image_id}")

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
        logging.error(f"Error processing image {image_id}: {e}")


def process_label_job(ai_service: AIService, redis_service: RedisService):
    with ThreadPoolExecutor() as executor:
        while not stop_event.is_set():  # Check stop event to gracefully exit
            try:
                messages = redis_service.read_from_stream(
                    stream_name='image_label_stream',
                    group_name='image_label_group',
                    consumer_name='worker_1',
                    count=1,
                    block=0,
                    start_id='>'
                )

                for stream, entries in messages:
                    for entry_id, fields in entries:
                        executor.submit(process_message, ai_service,
                                        redis_service, entry_id, fields)

            except Exception as e:
                logging.error(f"Error reading from Redis stream: {e}")


def process_pending_label_job(ai_service: AIService, redis_service: RedisService):
    while not stop_event.is_set():  # Check stop event to gracefully exit
        try:
            # Get pending messages
            pending_info = redis_service.client.xpending(
                'image_label_stream', 'image_label_group'
            )

            if pending_info['pending'] == 0:
                logging.info(
                    "No more pending messages in the stream. Stopping processing.")
                break

            # Get the list of pending messages
            pending_messages = redis_service.client.xpending_range(
                'image_label_stream', 'image_label_group', '-', '+', pending_info['pending']
            )

            for message in pending_messages:
                entry_id = message['message_id']

                # Read the message from the stream
                messages = redis_service.client.xrange(
                    'image_label_stream', min=entry_id, max=entry_id)

                # Process each message
                for message_data in messages:
                    entry_id, fields = message_data
                    print(f"Processing pending message: {entry_id}")
                    print(f"Fields: {fields}")

                    # Process the message (this will also handle ack_stream)
                    process_message(ai_service, redis_service,
                                    entry_id, fields)

            # Exit after processing all pending messages
            break

        except Exception as e:
            logging.error(f"Error processing pending messages: {e}")
            break


# Thread management functions
def start_stream_processors(ai_service: AIService, redis_service: RedisService):
    global old_stream_thread, new_stream_thread
    if old_stream_thread is None and new_stream_thread is None:
        # Start processing the streams in separate threads
        old_stream_thread = threading.Thread(
            target=process_pending_label_job, args=(ai_service, redis_service))
        new_stream_thread = threading.Thread(
            target=process_label_job, args=(ai_service, redis_service))

        old_stream_thread.daemon = True
        new_stream_thread.daemon = True

        old_stream_thread.start()
        new_stream_thread.start()


def stop_stream_processors():
    global old_stream_thread, new_stream_thread
    stop_event.set()  # Set stop event to signal threads to stop

    if old_stream_thread:
        old_stream_thread.join()  # Wait for the old stream thread to finish
        old_stream_thread = None

    if new_stream_thread:
        new_stream_thread.join()  # Wait for the new stream thread to finish
        new_stream_thread = None
