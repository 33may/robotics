from omni.isaac.kit import SimulationApp

# config dict для Isaac Sim
config = {
    "headless": True,
    "anti_aliasing": 0,
    "width": 1280,
    "height": 720,
    # ключевое: схема конфигурации такая же, как CLI-параметры, только в JSON/dict
    "/app/livestream/enabled": True,
    "/app/livestream/publicEndpointAddress": "100.115.105.111",
    "/app/livestream/port": 49100,
}

simulation_app = SimulationApp(config)
