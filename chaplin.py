import cv2
import time
# Removed Ollama dependency
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import os
from pynput import keyboard
import asyncio
from step4_llm_layer import SentenceGenerator


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:
    def __init__(self):
        self.vsr_model = None

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # Initialize core LLM logic (Gemini)
        self.sentence_generator = SentenceGenerator()

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None  # will be created in async loop
        self._init_async_resources()

        # setup global hotkey for toggling recording with option/alt key
        self.hotkey = keyboard.GlobalHotKeys({
            '<alt>': self.toggle_recording
        })
        self.hotkey.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    def toggle_recording(self):
        # toggle recording when alt/option key is pressed
        self.recording = not self.recording

    async def correct_output_async(self, output, sequence_num):
        # Use SentenceGenerator (Gemini) to refine the text
        # Run in thread to avoid blocking the async loop
        try:
            tokens = output.split()
            # Dummy confidence scores since VSR model outputs text
            scores = [1.0] * len(tokens)
            
            result = await asyncio.to_thread(
                self.sentence_generator.generate_sentence, 
                tokens, 
                scores
            )
            corrected_text = result.get('sentence', output)
            
            # Ensure it ends with punctuation if not present
            corrected_text = corrected_text.strip()
            if corrected_text and corrected_text[-1] not in ['.', '?', '!']:
                corrected_text += '.'
            
            # Add space for next sentence
            corrected_text += ' '
            
        except Exception as e:
            print(f"LLM Error: {e}")
            corrected_text = output + " "

        # wait until it's this task's turn to type
        async with self.typing_condition:
            while self.next_sequence_to_type != sequence_num:
                await self.typing_condition.wait()

            # this task's turn to type the corrected text
            self.kbd_controller.type(corrected_text)

            # increment sequence and notify next task
            self.next_sequence_to_type += 1
            self.typing_condition.notify_all()

        return corrected_text

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)

        # print the raw output to console
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n")

        # assign sequence number for this task
        sequence_num = self.current_sequence
        self.current_sequence += 1

        # start the async LLM correction (non-blocking) with sequence number
        asyncio.run_coroutine_threadsafe(
            self.correct_output_async(output, sequence_num),
            self.loop
        )

        # return immediately without waiting for correction
        return {
            "output": output,
            "video_path": video_path
        }

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(0)

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()

        futures = []
        output_path = ""
        out = None
        frame_count = 0

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                current_time = time.time()

                # conditional ensures that the video is recorded at the correct frame rate
                if current_time - last_frame_time >= self.frame_interval:
                    ret, frame = cap.read()
                    if ret:
                        # frame compression
                        encode_param = [
                            int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                        _, buffer = cv2.imencode('.jpg', frame, encode_param)
                        compressed_frame = cv2.imdecode(
                            buffer, cv2.IMREAD_GRAYSCALE)

                        if self.recording:
                            if out is None:
                                import tempfile
                                temp_dir = tempfile.gettempdir()
                                output_path = os.path.join(temp_dir, f"{self.output_prefix}_{time.time_ns() // 1_000_000}.mp4")
                                out = cv2.VideoWriter(
                                    output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    self.fps,
                                    (frame_width, frame_height),
                                    False  # isColor
                                )

                            out.write(compressed_frame)

                            # circle to indicate recording, only appears in the window and is not present in video saved to disk
                            cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)

                            frame_count += 1
                        # check if not recording AND video is at least 2 seconds long
                        elif not self.recording and frame_count > 0:
                            if out is not None:
                                out.release()
                                out = None

                            # only run inference if the video is at least 2 seconds long
                            if frame_count >= self.fps * 2:
                                futures.append(self.executor.submit(
                                    self.perform_inference, output_path))
                            else:
                                if os.path.exists(output_path):
                                    os.remove(output_path)

                            frame_count = 0

                        last_frame_time = current_time

                        # display the frame in the window
                        cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))

                # ensures that videos are handled in the order they were recorded
                for fut in futures:
                    if fut.done():
                        result = fut.result()
                        # once done processing, delete the video with the video path
                        os.remove(result["video_path"])
                        futures.remove(fut)
                    else:
                        break

        finally:
            # release everything
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            # cleanup any leftover temp files in the temp directory
            import tempfile
            temp_dir = tempfile.gettempdir()
            for file in os.listdir(temp_dir):
                if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass

            # stop global hotkey listener
            self.hotkey.stop()

            # stop async event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.async_thread.shutdown(wait=True)

            # shutdown executor
            self.executor.shutdown(wait=True)
