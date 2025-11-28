import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    # 1. Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # 2. Iterate through Images from attachments, download them and then save here
        for i, attachment in enumerate(attachments):
            if attachment.url:
                # Download the image from the bucket
                image_data = await bucket_client.get_file(attachment.url)
                
                # Create a filename with timestamp to avoid overwriting
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}_{i}.png"
                
                # Save the image locally
                with open(filename, 'wb') as f:
                    f.write(image_data)
                
                # 3. Print confirmation that image has been saved locally
                print(f"Image saved locally: {filename}")


def start() -> None:
    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="dall-e-3",
        api_key=API_KEY
    )
    
    # 2. Generate image for "Sunny day on Bali"
    message = Message(
        role=Role.USER,
        content="Sunny day on Bali"
    )
    
    # 4. Try to configure the picture for output via `custom_fields` parameter
    # DALL-E-3 supports size, quality, and style parameters
    dalle_custom_fields = {
        "size": Size.square,
        "quality": Quality.hd,
        "style": Style.vivid
    }
    
    print("=== Generating image with DALL-E-3 ===")
    response = client.get_completion(messages=[message], custom_fields=dalle_custom_fields)
    
    # 3. Get attachments from response and save generated message
    if response.custom_content and response.custom_content.attachments:
        print(f"\nGenerated {len(response.custom_content.attachments)} image(s)")
        asyncio.run(_save_images(response.custom_content.attachments))
    else:
        print("No images were generated in the response")
    
    # 5. Test it with the 'imagegeneration@005' (Google image generation model)
    print("\n=== Generating image with Google imagegeneration@005 ===")
    google_client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="imagegeneration@005",
        api_key=API_KEY
    )
    
    # Google's imagegeneration model has different parameters
    # Using empty custom_fields or Google-specific parameters
    google_response = google_client.get_completion(messages=[message])
    
    if google_response.custom_content and google_response.custom_content.attachments:
        print(f"\nGenerated {len(google_response.custom_content.attachments)} image(s)")
        asyncio.run(_save_images(google_response.custom_content.attachments))
    else:
        print("No images were generated in the response")


start()
