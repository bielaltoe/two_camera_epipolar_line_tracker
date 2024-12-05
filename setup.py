from setuptools import setup, find_packages


setup(
    name="two_view_tracker",
    version="0.0.2",
    description="",
    url="https://github.com/bielaltoe/two_camera_epipolar_line_tracker",
    author="bielaltoe",
    license="MIT",
    packages=find_packages("."),
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "two-view-tracker=two_view_tracker.main:main",
        ],
    },
    zip_safe=False,
    install_requires=[
        "numpy==2.1.3",
        "opencv-python==4.10.0.84",
        "opencv-python-headless==4.10.0.84",
        "ultralytics==8.3.40",
    ],
)
