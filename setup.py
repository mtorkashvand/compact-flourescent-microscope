import setuptools

requirements = [
    'docopt',
    'numpy',
    'pyzmq'
]

console_scripts = [
    'cfm_client=cfm.zmq.client:main',
    'cfm_client_with_gui=cfm.zmq.client_with_gui:main',
    'cfm_forwarder=cfm.zmq.forwarder:main',
    'cfm_hub=cfm.devices.hub_relay:main',
    'cfm_publisher=cfm.zmq.publisher:main',
    'cfm_server=cfm.zmq.server:main',
    'cfm_subscriber=cfm.zmq.subscriber:main',
    'cfm_logger=cfm.devices.logger:main',
    'cfm_displayer=cfm.devices.displayer:main',
    'cfm_data_hub=cfm.devices.data_hub:main',
    'cfm_writer=cfm.devices.writer:main',
    'cfm_processor=cfm.devices.processor:main',
    'cfm_commands=cfm.devices.commands:main',
    'cfm_tracker=cfm.devices.tracker:main',
    'flir_camera=cfm.devices.flir_camera:main',
    'xinput_pub=cfm.devices.xinput_pub:main',
    'cfm=cfm.system.cfm:main',
    'cfm_with_gui=cfm.system.cfm_with_gui:main',
    'cfm_teensy_commands=cfm.devices.teensy_commands:main',
]

setuptools.setup(
    name="cfm",
    version="0.1.0",
    author="Mahdi Torkashvand, Sina Rasouli",
    author_email="mmt.mahdi@gmail.com, rasoolibox193@gmail.com",
    description="Software to operate compact fluorescent microscope (CFM).",
    url="https://github.com/mtorkashvand/compact-flourescent-microscope",
    project_urls={
        "Bug Tracker": "https://github.com/mtorkashvand/compact-flourescent-microscope/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    entry_points={
        'console_scripts': console_scripts
    },
    packages=['cfm'],
    python_requires=">=3.6",
)
