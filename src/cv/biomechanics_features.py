import numpy as np


class BiomechanicsFeatureExtractor:
    """
    Extract biomechanical features from pose landmarks.
    """

    def __init__(self):
        pass

    def compute_angle(self, a, b, c):
        """
        Compute angle ABC from 3 points.
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        )

        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def extract_joint_angles(self, landmarks):

        features = {}

        # Example landmark indices from MediaPipe
        # hip=23, knee=25, ankle=27

        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]

        knee_angle = self.compute_angle(hip, knee, ankle)

        features["knee_angle"] = knee_angle

        return features

    def compute_symmetry(self, left_value, right_value):

        symmetry_index = abs(left_value - right_value) / (
            (left_value + right_value) / 2 + 1e-8
        )

        return symmetry_index

    def movement_smoothness(self, joint_series):

        # jerk approximation
        velocity = np.diff(joint_series)
        acceleration = np.diff(velocity)
        jerk = np.diff(acceleration)

        smoothness = np.mean(np.abs(jerk))

        return smoothness

    def extract_features_from_sequence(self, landmark_sequence):

        knee_angles = []

        for landmarks in landmark_sequence:

            joint_features = self.extract_joint_angles(landmarks)

            knee_angles.append(joint_features["knee_angle"])

        knee_angles = np.array(knee_angles)

        features = {}

        features["knee_angle_mean"] = np.mean(knee_angles)
        features["knee_angle_std"] = np.std(knee_angles)
        features["knee_angle_min"] = np.min(knee_angles)
        features["knee_angle_max"] = np.max(knee_angles)

        features["movement_smoothness"] = self.movement_smoothness(knee_angles)

        return features