import moviepy.editor as mp
import cv2
import numpy as np


def apply_vignette(frame):
    height, width = frame.shape[:2]
    center = (int(width / 2), int(height / 2))  # Convert to integers

    # Create a mask with a circular gradient
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, int(min(width, height) * 0.7), 255, -1, cv2.LINE_AA)

    # Apply Gaussian blur to the mask
    mask_blurred = cv2.GaussianBlur(mask, (0, 0), 100)

    # Convert mask to BGR and to the same data type as frame
    mask_bgr = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR)
    mask_bgr = mask_bgr.astype(frame.dtype)

    # Apply the vignette effect
    vignette = cv2.multiply(frame, mask_bgr / 255, dtype=cv2.CV_8U)
    return vignette

def apply_blur(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def apply_glitch(frame):
    # Create a copy of the frame
    frame_copy = frame.copy()

    # Apply horizontal shift
    shift = 10
    frame_copy[:, :-shift] = frame[:, shift:]

    # Apply vertical shift
    shift = 5
    frame_copy[:-shift, :] = frame[shift:, :]

    # Add noise
    noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
    glitched_frame = cv2.add(frame_copy, noise)

    return glitched_frame

def apply_combined_effect(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    combined_frame = np.where(gray_frame_bgr > 0, gray_frame_bgr, blurred_frame)
    return combined_frame

def apply_edge_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sepia(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_frame = cv2.transform(frame, sepia_filter)
    sepia_frame = np.clip(sepia_frame, 0, 255)
    return sepia_frame



# Load the video
input_video_path = 'task 3 video.mp4'
clip = mp.VideoFileClip(input_video_path)

# Define segments and effects
segments = [
    (5, 10, apply_vignette),
    (38, 42, apply_blur),
    (130, 140, apply_combined_effect),
    (206, 209, apply_edge_detection),
    (209, 213, apply_sepia),
    (213, 214, apply_glitch)
]

processed_clips = []
for start, end, effect_function in segments:
    segment = clip.subclip(start, end)
    processed_segment = segment.fl_image(effect_function)
    processed_clips.append(processed_segment)

# Concatenate the processed segments
final_clip = mp.concatenate_videoclips(processed_clips)

# Match the duration with the original video to prevent freezing
final_clip = final_clip.set_duration(sum([end-start for start, end, _ in segments]))
# # Concatenate the processed segments
# final_clip = mp.concatenate_videoclips(processed_clips)

# Save the processed video without audio
final_output_path_no_audio = 'final_output.mp4'
final_clip.write_videofile(final_output_path_no_audio, codec='libx264')

# # Define the path to the audio file
# audio_file = 'sound.mp3'
#
# # Check if the file exists
# if not os.path.exists(audio_file):
#     raise FileNotFoundError(f"The audio file {audio_file} does not exist. Please check the path.")
#
# # Load the audio file
# audio = mp.AudioFileClip(audio_file)
#
# # Set the audio to the processed video
# final_clip_with_audio = final_clip.set_audio(audio)
#
# # Save the final video with sound
# final_output_path_with_audio = 'final_output_with_audio.mp4'
# final_clip_with_audio.write_videofile(final_output_path_with_audio, codec='libx264')
