import discord
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
import pytz
from discord.ext import commands

nest_asyncio.apply()

# 토큰 & ID 설정
TOKEN = ""
CHANNEL_ID = 1267722917333172255

KST = pytz.timezone('Asia/Seoul')

notify_times = {
    "09:00:00": "수업 시작 시간 09:00 입니다. QR 코드를 찍고 출석체크를 완료해주세요!",
    "11:00:00": "실습 시작 시간 11:00 입니다. Zep 로그인 해주세요!",
    "11:50:00": "점심 시작 시간 11:50 입니다. 점심 맛있게 드세요~.",
    "13:00:00": "점심 종료 시간 13:00입니다. 자리로 돌아와 오후 공부를 시작해주세요!!.",
    "18:00:00": "수업 종료 시간 18:00 입니다. 오늘도 고생하셨습니다. QR 코드 찍고 퇴실 완료 잊지말아주세요!"
}

class Reminder(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.reminders = []
        self.bg_task = self.bot.loop.create_task(self.send_message_at_specific_times())

    @commands.Cog.listener()
    async def on_ready(self):
        print(f'Logged in as {self.bot.user}')


    @commands.command(name = "add_reminder")
    async def add_reminder(self, ctx, event_name:str, event_time:str, reminder_minutes: int =10):
        event_time_dt = datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S')
        reminder_time = event_time_dt - timedelta(minutes=reminder_minutes)

        self.reminders.append({
            "event_name": event_name,
            "event_time": event_time_dt,
            "reminder_time": reminder_time,
            "sending_check": False
        })

        await ctx.send(f"Reminder added : {event_name} at {event_time}")

    async def send_message_at_specific_times(self):
        await self.bot.wait_until_ready()
        channel = self.bot.get_channel(CHANNEL_ID)

        while not self.bot.is_closed():
            now = datetime.now(KST)

            # 오늘 알림 시간 설정 - 시간 문자열을 time 객체로 변환, combine으로 현재 날짜를 추가하여 시간 객체와 합쳐 datetime 생성
            today_times = {KST.localize(datetime.combine(now.date(), datetime.strptime(time_str, "%H:%M:%S").time())): msg for time_str, msg in notify_times .items()}

            future_times = {time: msg for time, msg in today_times.items() if time > now}

            if not future_times:
                next_time = min(today_times.keys()) + timedelta(minutes=1)
                message = today_times[min(today_times.keys())]

            else:
                next_time = min(future_times.keys())
                message = future_times[next_time]

            wait_time = (next_time - now).total_seconds()
            await asyncio.sleep(wait_time)

            await channel.send(f'{message} : {next_time.strftime("%Y-%m-%d %H:%M:%S")}')


            # 리마인더 체크 & 업데이트
            for reminder in self.reminders:
                if reminder['reminder_time'] <= now < reminder['event_time'] and not reminder['sending_check']:
                    await channel.send(
                        f"Reminder: {reminder['event_name']} at {reminder['event_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    reminder['sending_check'] = True  # 알림을 보낸 후 플래그를 업데이트합니다.


if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True

    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        print(f'Logged in as {bot.user}')
    async def setup():
        await bot.add_cog(Reminder(bot))

    async def main():
        async with bot:
            await setup()
            await bot.start(TOKEN)


    asyncio.run(main())
