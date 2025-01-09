class SQLAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    
    def generate_sql(self, understanding: dict) -> str:
        
        # 根据信息抽取结果生成对应 sql
        prompt = self.build_sql_prompt(understanding)
        sql = self.llm.generate(prompt)
        
        # TODO: 再次调用进行反思，优化生成的sql（包括表连接，条件等）

        
        return sql
    
    def build_sql_prompt(self, understanding: dict) -> str:
        conditions = []
        
           
        return ' AND '.join(conditions)
    
    def generate_sql_hint(self, question: str) -> str:
        sql_hints = ""
        sql_hints += ">>查询参考："

        if "近一个月最高价" in question:
            sql_hints += "查询近一个月最高价,你写的sql语句可以优先考虑表中已有字段HighPriceRM  近一月最高价(元)  "
        if "近一个月最低价" in question:
            sql_hints += "查询近一月最低价(元),你写的sql语句直接调用已有字段LowPriceRM"
        if "行业" in question and (
            "多少只" in question or "几个" in question or "多少个" in question
        ):
            sql_hints += """查询某行业某年数量 示例sql语句:SELECT count(*) as 风电零部件_2021
                    FROM AStockIndustryDB.LC_ExgIndustry
                    where ThirdIndustryName like '%风电零部件%' and year(InfoPublDate)=2021 and IfPerformed = 1;"""
        if ("年度报告" in question and "最新更新" in question) or "比例合计" in question:
            sql_hints += """特别重要一定注意，查询最新更新XXXX年年度报告，机构持有无限售流通A股数量合计InstitutionsHoldProp最多公司代码，优先使用查询sql语句，SELECT *
                                    FROM AStockShareholderDB.LC_StockHoldingSt
                                    WHERE date(EndDate) = 'XXXX-12-31'
                                    AND UpdateTime = (
                                        SELECT MAX(UpdateTime)
                                        FROM AStockShareholderDB.LC_StockHoldingSt
                                        WHERE date(EndDate) = 'XXXX-12-31'
                                    ) order by InstitutionsHoldings desc limit 1 ，XXXX代表问题查询年度，sql语句禁止出现group by InnerCode;

                                    查询最新更新XXXX年年度报告,公司机构持有无限售流通A股比例合计InstitutionsHoldProp是多少,优先使用查询sql语句，SELECT InstitutionsHoldProp
                                    FROM AStockShareholderDB.LC_StockHoldingSt
                                    WHERE date(EndDate) = 'XXXX-12-31'
                                    AND UpdateTime = (
                                        SELECT MAX(UpdateTime)
                                        FROM AStockShareholderDB.LC_StockHoldingSt
                                        WHERE date(EndDate) = 'XXXX-12-31'
                                    ) order by InstitutionsHoldings desc limit 1 ，XXXX代表问题查询年度，sql语句禁止出现group by InnerCode;"""

        if "新高" in question:
            sql_hints += """新高 要用AStockMarketQuotesDB.CS_StockPatterns现有字段
                
                查询今天是2021年01月01日，创近半年新高的股票有几只。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                        where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01;
                        判断某日 YY-MM-DD  InnerCode XXXXXX 是否创近一周的新高，查询结果1代表是,IfHighestHPriceRW字段可以根据情况灵活调整  SELECT   InnerCode,TradingDay,IfHighestHPriceRW  FROM AStockMarketQuotesDB.CS_StockPatterns
        where  date(TradingDay)='2021-12-20' and InnerCode = '311490'
                        
                        """
        if "成交额" in question and "平均" in question:
            sql_hints += """查询这家公司5日内平均成交额是多少。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                        where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01"""
            
        if "半年度报告" in question:
            sql_hints += """查询XXXX年半年度报告的条件为：year(EndDate) = XXXX and InfoSource='半年度报告'"""

        return sql_hints