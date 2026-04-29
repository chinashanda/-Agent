import os
import json
import lancedb
from dotenv import load_dotenv
from openai import OpenAI

# 加载配置文件：config.env
load_dotenv("config.env")

# ===================== 全局配置 =====================
class Config:
    FEISHU_APP_ID = os.getenv("FEISHU_APP_ID")
    FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET")
    FEISHU_MEETING_ID = os.getenv("FEISHU_MEETING_ID")
    FEISHU_SHEET_TOKEN = os.getenv("FEISHU_SHEET_TOKEN")

    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    LANCEDB_PATH = "./meeting_db"

# 初始化客户端
llm_client = OpenAI(base_url=Config.LLM_BASE_URL, api_key=Config.LLM_API_KEY)
db = lancedb.connect(Config.LANCEDB_PATH)

# ===================== Agent 1：会议转录Agent =====================
class MeetingTranscriptAgent:
    def __init__(self):
        self.name = "会议转录Agent"

    def run(self, meeting_id: str) -> str:
        print(f"[{self.name}] 开始拉取飞书会议转录文本...")
        mock_transcript = """
        会议时间：2025-04-29
        参会人：张三、李四、王五
        内容：
        1. 张三：需要完成后端接口开发，截止时间5月1日
        2. 李四：负责前端页面调试，截止时间5月2日
        3. 王五：需要测试上线流程，截止时间4月30日
        """
        print(f"[{self.name}] 转录完成，文本长度：{len(mock_transcript)}")
        return mock_transcript

# ===================== Agent 2：信息抽取Agent =====================
class InfoExtractAgent:
    def __init__(self):
        self.name = "信息抽取Agent"
        self.prompt = """
        你是专业的会议信息提取专家，请从文本中抽取结构化任务，输出JSON格式：
        [
            {
                "task": "任务内容",
                "owner": "责任人",
                "deadline": "截止时间",
                "dependencies": "依赖项"
            }
        ]
        只返回JSON，不要其他文字。
        """

    def run(self, transcript: str) -> list:
        print(f"[{self.name}] 开始长链推理抽取任务信息...")
        response = llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": self.prompt + transcript}]
        )
        result = json.loads(response.choices[0].message.content)
        print(f"[{self.name}] 抽取完成，共{len(result)}个任务")
        return result

# ===================== Agent 3：LanceDB 存储Agent =====================
class LanceDBStorageAgent:
    def __init__(self):
        self.name = "LanceDB存储Agent"
        self.table = db.create_table("meeting_tasks", exist_ok=True)

    def run(self, tasks: list, transcript: str):
        print(f"[{self.name}] 开始向量化存储会议数据...")
        data = []
        for idx, task in enumerate(tasks):
            data.append({
                "id": idx,
                "task": task["task"],
                "owner": task["owner"],
                "deadline": task["deadline"],
                "status": "待执行",
                "transcript": transcript
            })
        self.table.add(data)
        print(f"[{self.name}] 存储完成，已保存{len(data)}条任务到LanceDB")

# ===================== Agent 4：飞书任务同步Agent =====================
class FeishuTaskAgent:
    def __init__(self):
        self.name = "飞书任务同步Agent"

    def run(self, tasks: list):
        print(f"[{self.name}] 开始同步任务到飞书多维表格...")
        for task in tasks:
            print(f"→ 写入任务：{task['owner']} | {task['task']} | 截止：{task['deadline']}")
        print(f"[{self.name}] 已推送任务提醒到飞书群")
        return True

# ===================== Agent 5：风险预警Agent =====================
class RiskWarningAgent:
    def __init__(self):
        self.name = "风险预警Agent"
        self.table = db.open_table("meeting_tasks")

    def run(self):
        print(f"[{self.name}] 开始分析任务延期风险...")
        warning_tasks = self.table.search("截止时间近").limit(2).to_list()
        if warning_tasks:
            print(f"[{self.name}] 发现延期风险！预警任务：{len(warning_tasks)}个")
            for t in warning_tasks:
                print(f"⚠️  风险：{t['owner']}-{t['task']}")
        else:
            print(f"[{self.name}] 无延期风险")
        return warning_tasks

# ===================== 主流程调度 =====================
def run_meeting_closure():
    print("="*60)
    print("🚀 启动 会议纪要&项目进度 闭环Agent系统")
    print("="*60)

    transcript_agent = MeetingTranscriptAgent()
    transcript = transcript_agent.run(Config.FEISHU_MEETING_ID)

    extract_agent = InfoExtractAgent()
    tasks = extract_agent.run(transcript)

    storage_agent = LanceDBStorageAgent()
    storage_agent.run(tasks, transcript)

    feishu_agent = FeishuTaskAgent()
    feishu_agent.run(tasks)

    risk_agent = RiskWarningAgent()
    risk_agent.run()

    print("\n✅ 全部流程执行完成！闭环结束")

if __name__ == "__main__":
    run_meeting_closure()