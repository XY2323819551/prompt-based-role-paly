import os
import json
from dotenv import load_dotenv
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from tool_pool import BaseTool

class WeatherTool(BaseTool):
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get('OPENWEATHERMAP_API_KEY')
        if not api_key:
            raise ValueError('OPENWEATHERMAP_API_KEY environment variable is not set')
        wrapper = OpenWeatherMapAPIWrapper()
        self.tool = OpenWeatherMapQueryRun(api_wrapper=wrapper)
        super().__init__()
    
    def run(self, city):
        return self.tool.run(city)
    

def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    使用 OpenWeatherMap API 获取指定城市的当前天气信息，并返回格式化的结果。

    Args:
        city (str): 需要查询天气的城市名称，使用英文来表示城市的名称，比如：London、shanghai、shenzhen。

    Returns:
        dict: 包含天气信息的字典，具体包含以下字段：
            - status (str): 天气状况描述
            - wind (dict): 风力信息
                - speed (float): 风速（米/秒）
                - direction (int): 风向（角度）
            - humidity (int): 湿度百分比
            - temperature (dict): 温度信息
                - current (float): 当前温度（摄氏度）
                - high (float): 最高温度（摄氏度）
                - low (float): 最低温度（摄氏度）
                - feels_like (float): 体感温度（摄氏度）
            - cloud_cover (int): 云量百分比
            - rain (dict): 降雨信息（如果有）

    Raises:
        ValueError: 当城市名称无效或API调用失败时抛出。
    """
    tool = WeatherTool()
    weather_info = tool.run(city)
    
    # 格式化返回结果
    formatted_weather = {
        "status": weather_info.split("Detailed status: ")[1].split("\n")[0],
        "wind": {
            "speed": float(weather_info.split("Wind speed: ")[1].split(" m/s")[0]),
            "direction": int(weather_info.split("direction: ")[1].split("°")[0])
        },
        "humidity": int(weather_info.split("Humidity: ")[1].split("%")[0]),
        "temperature": {
            "current": float(weather_info.split("Current: ")[1].split("°C")[0]),
            "high": float(weather_info.split("High: ")[1].split("°C")[0]),
            "low": float(weather_info.split("Low: ")[1].split("°C")[0]),
            "feels_like": float(weather_info.split("Feels like: ")[1].split("°C")[0])
        },
        "cloud_cover": int(weather_info.split("Cloud cover: ")[1].split("%")[0])
    }
    
    # 如果有降雨信息，添加到结果中
    rain_info = weather_info.split("Rain: ")[1].split("\n")[0]
    if rain_info != "{}":
        formatted_weather["rain"] = eval(rain_info)
    else:
        formatted_weather["rain"] = {}
    
    return json.dumps(formatted_weather, ensure_ascii=False)


if __name__ == '__main__':
    print("================")
    # print(get_weather('London'))
    print(get_weather('shanghai'))
    print("================")
