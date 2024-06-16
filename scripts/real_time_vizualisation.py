import pygame
import numpy as np
import sounddevice as sd
import librosa
import joblib
import os
import random

model_path = 'models/'
models = [joblib.load(os.path.join(model_path, f'model_visual_param_{i}.pkl')) for i in range(5)]

pygame.init()

screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

width, height = pygame.display.get_surface().get_size()

pygame.display.set_caption("Reyna Deyna")
clock = pygame.time.Clock()

font_size = 20
font = pygame.font.Font(None, font_size)
matrix_chars = '･ﾟ✧R*E:Y･ﾟnꕥA*111:･'

def extract_features(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).mean(axis=1)
    tempo = librosa.beat.tempo(y=audio, sr=sr)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
    rms = librosa.feature.rms(y=audio)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0].mean()

    features = np.hstack([chroma, [tempo, spectral_centroid, spectral_bandwidth, rolloff, rms, zcr]])

    return features, spectral_centroid, spectral_bandwidth, rms

def get_gradient_color(base_color, intensity, max_intensity):
    factor = intensity / max_intensity
    baby_pink = (255, 182, 193)
    return (
        int(base_color[0] + (baby_pink[0] - base_color[0]) * factor),
        int(base_color[1] + (baby_pink[1] - base_color[1]) * factor),
        int(base_color[2] + (baby_pink[2] - base_color[2]) * factor),
    )

def draw_celtic_knot(screen, base_color, center_x, center_y, radius, angle_offset):
    num_segments = 12
    angle_step = 2 * np.pi / num_segments
    for i in range(num_segments):
        angle1 = angle_offset + i * angle_step
        angle2 = angle1 + angle_step / 2
        x1 = center_x + radius * np.cos(angle1)
        y1 = center_y + radius * np.sin(angle1)
        x2 = center_x + radius * np.cos(angle2)
        y2 = center_y + radius * np.sin(angle2)
        color = get_gradient_color(base_color, i, num_segments)
        pygame.draw.line(screen, color, (int(x1), int(y1)), (int(x2), int(y2)), 3)
        pygame.draw.circle(screen, color, (int(x1), int(y1)), 5)

def draw_dynamic_heart(screen, base_color, center_x, center_y, angle_offset):
    num_points = 100
    angle_step = 2 * np.pi / num_points
    pulsation_factor = 20

    for i in range(num_points):
        angle = angle_offset + i * angle_step
        pulsation = np.sin(angle) * pulsation_factor

        x = center_x + 11 * 10 * (np.sin(angle) ** 3)
        y = center_y - 11 * (13 * np.cos(angle) - 5 * np.cos(2 * angle) - 2 * np.cos(3 * angle) - np.cos(4 * angle)) + pulsation

        color = get_gradient_color(base_color, i, num_points)

        pygame.draw.circle(screen, color, (int(x), int(y)), 2)

def draw_spiral_sun(screen, base_color, center_x, center_y, radius, angle_offset, bpm, rms):
    pulsating_radius = radius + (bpm / 60) * 150
    angle = angle_offset
    num_segments = 100
    for i in range(num_segments):
        color = get_gradient_color(base_color, i, num_segments)
        x = center_x + pulsating_radius * np.cos(angle)
        y = center_y + pulsating_radius * np.sin(angle)
        pygame.draw.line(screen, color, (center_x, center_y), (int(x), int(y)), 2)
        angle += 2 * np.pi / num_segments

def draw_complex_spiral(screen, base_color, center_x, center_y, angle_offset):
    angle_step = 0.1
    radius_step = 2
    for i in range(1, 100):
        color = get_gradient_color(base_color, i, 100)
        angle = angle_offset + i * angle_step
        r = i * radius_step
        x = center_x + int(r * np.cos(angle))
        y = center_y + int(r * np.sin(angle))
        pygame.draw.circle(screen, color, (x, y), 5)

def draw_circular_pattern(screen, base_color, center_x, center_y, radius, angle_offset):
    num_points = 100
    angle_step = 2 * np.pi / num_points
    for i in range(num_points):
        angle = angle_offset + i * angle_step
        r = i * radius * 0.05
        x = center_x + int(r * np.cos(angle))
        y = center_y + int(r * np.sin(angle))
        pygame.draw.circle(screen, base_color, (x, y), 5)

def draw_visualization(screen, visual_params, angle_offset, spectral_centroid, spectral_bandwidth, rms):
    screen.fill((0, 0, 0))

    base_color = (255, 255, 255)
    center_x = width // 2
    center_y = height // 2 + int((spectral_centroid - 2500) / 10)
    radius = int(visual_params[2] * width * 0.1 + rms * 100)
    angle_offset += visual_params[3] * 0.1

    if spectral_centroid < 2000:
        draw_celtic_knot(screen, base_color, center_x, center_y, radius, angle_offset)
    elif spectral_centroid < 3000:
        draw_dynamic_heart(screen, base_color, center_x, center_y, angle_offset)
    elif spectral_centroid < 4000:
        draw_spiral_sun(screen, base_color, center_x, center_y, radius, angle_offset, visual_params[0], rms)
    elif spectral_centroid < 5000:
        draw_complex_spiral(screen, base_color, center_x, center_y, angle_offset)
    else:
        draw_circular_pattern(screen, base_color, center_x, center_y, radius, angle_offset)

    baby_pink = (255, 182, 193)
    for i in range(len(matrix_rain_non_triggered)):
        for j in range(len(matrix_rain_non_triggered[i])):
            char, pos_y = matrix_rain_non_triggered[i][j]
            pos_x = i * font_size
            if pos_y > 0 and pos_y < height:
                color = get_gradient_color(baby_pink, pos_y, height)
                screen.blit(char, (pos_x, pos_y))
                char.set_alpha(100)
            matrix_rain_non_triggered[i][j] = (char, pos_y + 2)

            if pos_y >= height:
                matrix_rain_non_triggered[i][j] = (char, -font_size)

    pygame.display.flip()

def audio_callback(indata, frames, time, status):
    audio = indata[:, 0]
    sr = 44100
    features, spectral_centroid, spectral_bandwidth, rms = extract_features(audio, sr)
    features = features.reshape(1, -1)

    visual_params = [model.predict(features)[0] for model in models]

    global angle_offset
    angle_offset += visual_params[3] * 0.1

    draw_visualization(screen, visual_params, angle_offset, spectral_centroid, spectral_bandwidth, rms)

angle_offset = 0

matrix_rain_non_triggered = [[(font.render(random.choice(matrix_chars), True, (255, 192, 203)), random.randint(-height, 0)) for _ in range(height // font_size)] for _ in range(width // font_size)]
matrix_rain_triggered = [[(font.render(random.choice(matrix_chars), True, (255, 192, 203)), random.randint(-height, 0)) for _ in range(height // font_size)] for _ in range(width // font_size)]

stream = sd.InputStream(callback=audio_callback, blocksize=2048, samplerate=44100, channels=1)

fullscreen_mode = False

def toggle_fullscreen():
    global fullscreen_mode
    if fullscreen_mode:
        pygame.display.set_mode((width, height))
        fullscreen_mode = False
    else:
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        fullscreen_mode = True

with stream:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    toggle_fullscreen()
        clock.tick(60)

pygame.quit()
