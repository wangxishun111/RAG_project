"""AutoGen Swarm 多 Agent 旅行规划示例。

运行前请确保已安装 autogen-agentchat、autogen-ext、python-dotenv、requests，
并在 .env 中配置 DEEPSEEK_API_KEY 与 DEEPSEEK_BASE_URL。
"""

import asyncio
import os
from typing import Dict, List, Tuple

import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


CITY_ATTRACTIONS: Dict[str, List[str]] = {
    "北京": ["故宫", "长城", "颐和园", "天坛"],
    "上海": ["东方明珠", "外滩", "豫园", "南京路步行街"],
    "广州": ["白云山", "珠江夜游", "广州塔", "陈家祠"],
    "深圳": ["深圳湾公园", "东部华侨城", "大梅沙海滨公园", "世界之窗"],
    "成都": ["大熊猫繁育研究基地", "宽窄巷子", "锦里古街", "都江堰"],
    "西安": ["兵马俑", "大雁塔", "华清池", "钟鼓楼", "西安城墙", "大唐不夜城"],
    "杭州": ["西湖", "灵隐寺", "宋城", "京杭大运河"],
}

CITY_FOODS: Dict[str, List[str]] = {
    "北京": ["北京烤鸭", "炸酱面", "豆汁儿", "卤煮"],
    "上海": ["生煎包", "小笼包", "本帮红烧肉", "葱油拌面"],
    "广州": ["早茶", "肠粉", "云吞面", "烧鹅"],
    "深圳": ["光明乳鸽", "沙井蚝", "客家酿豆腐", "椰子鸡"],
    "成都": ["火锅", "串串香", "担担面", "钟水饺"],
    "西安": ["肉夹馍", "羊肉泡馍", "凉皮", "biangbiang面", "甑糕"],
    "杭州": ["西湖醋鱼", "东坡肉", "龙井虾仁", "片儿川"],
}

ROUTE_TIMES: Dict[Tuple[str, str], str] = {
    ("钟楼", "兵马俑"): "约 70-90 分钟",
    ("兵马俑", "华清池"): "约 20-30 分钟",
    ("华清池", "大雁塔"): "约 70-90 分钟",
    ("钟楼", "大雁塔"): "约 25-35 分钟",
    ("大雁塔", "大唐不夜城"): "步行约 10-15 分钟",
    ("钟楼", "西安城墙"): "约 10-20 分钟",
    ("西安城墙", "回民街"): "约 10-20 分钟",
    ("回民街", "钟楼"): "步行约 10-15 分钟",
}

BUDGET_LEVELS: Dict[str, Dict[str, int]] = {
    "low": {"hotel": 180, "food": 80, "transport": 40, "ticket": 120},
    "medium": {"hotel": 350, "food": 150, "transport": 80, "ticket": 220},
    "high": {"hotel": 800, "food": 300, "transport": 180, "ticket": 350},
}


async def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        desc = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        return f"当前{city}天气：{desc}，温度{temp}℃，体感温度{feels_like}℃。"
    except Exception as exc:
        return f"查询{city}天气失败：{exc}。请在最终计划中提醒用户出发前再次确认天气。"


async def get_attractions(city: str) -> str:
    """获取指定城市的推荐景点。"""
    attractions = CITY_ATTRACTIONS.get(city)
    if not attractions:
        return f"暂未收录{city}的固定景点清单，请根据通用旅行经验给出建议。"
    return f"{city}推荐景点：{'、'.join(attractions)}。"


async def get_foods(city: str) -> str:
    """获取指定城市的特色美食。"""
    foods = CITY_FOODS.get(city)
    if not foods:
        return f"暂未收录{city}的固定美食清单，请根据通用旅行经验给出建议。"
    return f"{city}特色美食：{'、'.join(foods)}。"


async def estimate_budget(city: str, days: int, budget_level: str = "medium") -> str:
    """估算住宿、餐饮、市内交通和门票预算。"""
    level = budget_level if budget_level in BUDGET_LEVELS else "medium"
    cost = BUDGET_LEVELS[level]
    hotel_days = max(days - 1, 1)
    hotel_total = hotel_days * cost["hotel"]
    food_total = days * cost["food"]
    transport_total = days * cost["transport"]
    ticket_total = days * cost["ticket"]
    total = hotel_total + food_total + transport_total + ticket_total

    return (
        f"{city}{days}天{level}档预算估算："
        f"住宿约{hotel_total}元，餐饮约{food_total}元，"
        f"市内交通约{transport_total}元，门票约{ticket_total}元，"
        f"合计约{total}元；不含往返大交通和购物。"
    )


async def estimate_route_time(start: str, end: str) -> str:
    """模拟两个地点之间的交通时间。"""
    duration = ROUTE_TIMES.get((start, end)) or ROUTE_TIMES.get((end, start))
    if duration is None:
        duration = "约 30-60 分钟，建议以地图实时导航为准"
    return f"{start} 到 {end} 的交通时间：{duration}。"


def create_model_client() -> OpenAIChatCompletionClient:
    """创建兼容 OpenAI API 的模型客户端。"""
    load_dotenv()
    return OpenAIChatCompletionClient(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
    )


def create_travel_team(model_client: OpenAIChatCompletionClient) -> Swarm:
    """创建支持动态交接的 Swarm 旅行规划团队。"""
    weather_tool = FunctionTool(
        func=get_weather,
        description="查询指定城市的实时天气信息。参数 city 是中文城市名。",
    )
    attraction_tool = FunctionTool(
        func=get_attractions,
        description="查询指定城市的推荐景点。参数 city 是中文城市名。",
    )
    food_tool = FunctionTool(
        func=get_foods,
        description="查询指定城市的特色美食。参数 city 是中文城市名。",
    )
    budget_tool = FunctionTool(
        func=estimate_budget,
        description=(
            "估算旅行预算。参数 city 是中文城市名，days 是旅行天数，"
            "budget_level 可选 low、medium、high。"
        ),
    )
    route_time_tool = FunctionTool(
        func=estimate_route_time,
        description="估算两个景点或地点之间的交通时间。参数 start 和 end 是地点名。",
    )

    requirement_analyst = AssistantAgent(
        name="requirement_analyst",
        model_client=model_client,
        handoffs=["destination_researcher"],
        system_message=(
            "你是需求分析 Agent，是 Swarm 的起点。请从用户请求中解析旅行需求，并输出："
            "目的地城市、旅行天数、预算档位、兴趣偏好、节奏偏好、同行人信息、缺失信息和默认假设。"
            "预算档位只能归纳为 low、medium、high；如果用户未说明预算，默认 medium；"
            "如果用户未说明偏好，默认兼顾景点、美食和天气。"
            "完成需求分析后，必须交接给 destination_researcher。不要输出 Finish。"
        ),
    )

    destination_researcher = AssistantAgent(
        name="destination_researcher",
        model_client=model_client,
        tools=[weather_tool, attraction_tool],
        handoffs=["food_expert"],
        system_message=(
            "你是目的地信息调研 Agent。请基于需求分析结果确定城市，调用工具查询天气和景点，"
            "并把结果整理成简洁要点。完成后必须交接给 food_expert。不要输出 Finish。"
        ),
    )

    food_expert = AssistantAgent(
        name="food_expert",
        model_client=model_client,
        tools=[food_tool],
        handoffs=["budget_agent"],
        system_message=(
            "你是当地美食 Agent。请根据目的地推荐特色美食、用餐区域和饮食提醒，必要时调用美食工具。"
            "完成后必须交接给 budget_agent。不要输出 Finish。"
        ),
    )

    budget_agent = AssistantAgent(
        name="budget_agent",
        model_client=model_client,
        tools=[budget_tool],
        handoffs=["itinerary_planner"],
        system_message=(
            "你是预算规划 Agent。请根据需求分析中的城市、天数和预算档位，调用预算工具估算费用。"
            "如果缺少信息，请采用需求分析 Agent 的默认假设。完成后必须交接给 itinerary_planner。"
            "不要输出 Finish。"
        ),
    )

    itinerary_planner = AssistantAgent(
        name="itinerary_planner",
        model_client=model_client,
        tools=[route_time_tool],
        handoffs=["reviewer_agent"],
        system_message=(
            "你是行程规划与修订 Agent。你需要综合需求、天气、景点、美食和预算信息生成旅行计划。"
            "如果这是第一次收到任务，请输出 PLAN_VERSION: DRAFT，并生成初版计划；"
            "如果你收到 reviewer_agent 的 REVIEW_STATUS: NEED_REVISION，请输出 PLAN_VERSION: REVISED，"
            "并逐条回应审核意见后重新规划。规划或修订时，应调用路线时间工具估算关键路段交通时间。"
            "完成初稿或修订稿后，必须交接给 reviewer_agent。不要输出 Finish。"
        ),
    )

    reviewer_agent = AssistantAgent(
        name="reviewer_agent",
        model_client=model_client,
        handoffs=["itinerary_planner", "final_writer"],
        system_message=(
            "你是旅行计划审核 Agent。请检查 itinerary_planner 的最新方案是否合理，重点检查："
            "时间是否过满、路线是否绕路、预算是否缺失或不匹配、天气风险是否覆盖、"
            "美食和休息时间是否安排、是否存在不可执行或过于笼统的描述。"
            "你必须在第一行输出 REVIEW_STATUS: PASS 或 REVIEW_STATUS: NEED_REVISION。"
            "如果问题会明显影响可执行性，输出 REVIEW_STATUS: NEED_REVISION，并列出必须修改的问题，"
            "然后交接给 itinerary_planner 返工。"
            "如果方案已经可执行，输出 REVIEW_STATUS: PASS，并交接给 final_writer。"
            "如果已经看到 PLAN_VERSION: REVISED，除非仍有严重问题，否则应倾向于 PASS，避免无限返工。"
            "不要输出 Finish。"
        ),
    )

    final_writer = AssistantAgent(
        name="final_writer",
        model_client=model_client,
        system_message=(
            "你是最终旅行计划撰写 Agent。只有当 reviewer_agent 输出 REVIEW_STATUS: PASS 后，你才应该输出最终计划。"
            "请基于最新通过审核的计划，输出一份清晰、可执行、按天分段的最终旅行计划。"
            "必须包含：需求摘要、每日路线、关键路段交通时间、景点安排、餐饮建议、预算估算、"
            "天气提醒和备选方案。最终回答末尾必须写 Finish。"
        ),
    )

    return Swarm(
        participants=[
            requirement_analyst,
            destination_researcher,
            food_expert,
            budget_agent,
            itinerary_planner,
            reviewer_agent,
            final_writer,
        ],
        termination_condition=TextMentionTermination("Finish") | MaxMessageTermination(30),
    )


async def main() -> None:
    """运行 Swarm 多 Agent 旅行规划示例。"""
    model_client = create_model_client()
    try:
        travel_team = create_travel_team(model_client)
        task = "我计划周末去西安旅行 3 天，高预算，不喜欢历史景点，不喜欢面食，请帮我制定行程。"
        await Console(travel_team.run_stream(task=task))
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
  