The most dangerous thing online today might be this

MIT has put an entire quantitative finance course online for free: more than twenty lectures, with MIT math professors teaching the math and Morgan Stanley practitioners explaining how to use it on the trading desk

I turned the methods from this course that can be used in trading into a Claude skill and released it completely free

Install this one skill, and Claude Fable 5.2 becomes a trading bot that never stops: it researches the market for you overnight, using the math from the MIT course to find strategies that can make money

First, understand what it is:

> What the course teaches is the math quantitative traders use every day
> Methods from the course like volatility, VaR, and cointegration go into a skill, and Claude runs them directly on real data
> Works with stocks, ETFs, and crypto. The data is free, and no API key is needed
> No coding required. Just tell Claude in plain language
> This is not an MIT product. It was built from MIT’s publicly available free course

Starting from scratch with Fable 5.2:

1 Paste the repository link into Claude Code and tell Claude, “Install the skill.” That’s it. No signup, no API key, no coding. Your own AI trading desk is ready

2 Before bed, enter “/mit-trader research.” It will test each of these strategy types—momentum, mean reversion, breakout, and cointegrated pairs—one by one using years of real stock, ETF, and crypto data

3 Every strategy has to win in years it has never seen (rolling forward), pass the luck test, and still be alive after real trading costs. Otherwise, it’s straight to REJECTED

4 Every morning, enter “/mit-trader today.” First check whether any surviving strategies trigger today, then use the GARCH volatility model (Lecture 9) to place stop-losses beyond the reach of normal price swings

5 Next, run a Monte Carlo simulation of each trade tens of thousands of times, then use Ed Thorp’s Kelly formula to calculate how much to bet. If the result is zero, don’t bet a penny

6 It checks how much the entire account could lose on its worst day (VaR), cross-checks every limit you’ve set, and finally puts the prepared order ticket in front of you, waiting for your approval

7 By default, it only records a simulated ledger and doesn’t connect to a broker. If you don’t type “yes” for that trade, it won’t buy a single thing

8 Set the schedule just once: for example, run research at 2 a.m. every night, run today once after the market opens on weekdays, and check crypto every 2 hours. That way, it runs around the clock

9 If you don’t understand a number, ask “/mit-trader explain Kelly until a 15-year-old can understand it.” It will explain in plain language and tell you which lecture or paper it comes from

Its most valuable quality is that it dares to say no: I ran it on real data today, and all 36 candidates were REJECTED. Not a single order ticket was produced