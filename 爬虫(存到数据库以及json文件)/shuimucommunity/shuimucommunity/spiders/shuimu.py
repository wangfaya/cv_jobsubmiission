# -*- coding: utf-8 -*-
import scrapy,re
from ..items import ShuimucommunityItem

class ShuimuSpider(scrapy.Spider):
    name = 'shuimu'
    allowed_domains = ['newsmth.net']
    url='http://www.newsmth.net/nForum/board/EnglishWorld?ajax&p='
    num=1

    start_urls = [url+str(num)]

    def start_requests(self):
        # 这个cookies_str是抓包获取的
        cookies_str = 'nforum-left=10000; left-index=00000001000; main[XWJOKE]=hoho; NFORUM=uaj9kg74q8kh40i4js662lqjm7; Hm_lvt_9c7f4d9b7c00cb5aba2c637c64a41567=1539567125,1539567208,1539572946,1540172650; main[UTMPUSERID]=wcxl123; main[UTMPKEY]=41014232; main[UTMPNUM]=8381; main[PASSWORD]=%2509%2524%2515R%2501%250BJMe%250F%2503%250CWrq%2515W%2519%2501%2500%252Cd%255C_; Hm_lpvt_9c7f4d9b7c00cb5aba2c637c64a41567=1540173617'  # 抓包获取
        # 将cookies_str转换为cookies_dict
        cookies_dict = {i.split('=')[0]: i.split('=')[1] for i in cookies_str.split('; ')}
        yield scrapy.Request(
            self.start_urls[0],
            callback=self.parse,
            cookies=cookies_dict
        )



    def parse(self, response):
        for each in response.xpath("//div[@class='b-content']/table/tbody/tr"):
            item=ShuimucommunityItem()
            item['name']=each.xpath("./td[2]/a/text()").extract()[0]
            item['email']='http://www.newsmth.net'+each.xpath('./td[2]/a/@href').extract()[0]+'?ajax'
            item['postedtime']=each.xpath("./td[3]/text()").extract()[0]
            item['author']=each.xpath("./td[4]/a/text()").extract()[0]
            item['replynum']=each.xpath("./td[7]/text()").extract()[0]
            item['replytime']=each.xpath("./td[8]/a/text()").extract()[0]
            item['replyauthor']=each.xpath("./td[9]/a/text()").extract()[0]
            # item['content']=each.xpath('./td[2]/a/@href').extract()[0]
            newulr=response.urljoin(item['email'])
            yield scrapy.Request(url=newulr,callback=self.content,meta={'item':item})

            # yield item

        if self.num<477:
            self.num+=1
        yield scrapy.Request(self.url+str(self.num),callback=self.parse)




    def content(self,response):
        item=response.meta['item']
        strings=""
        content_text=response.xpath("//tr[@class='a-body']/td[@class='a-content']/p/text()").extract()
        for str in content_text:
            strings+=str

        # html_content_text=response.text
        # content_text=re.findall("<p>(.*?)</p>",html_content_text)
        #
        #
        # content_str = ''
        # for i in content_text:
        #     content_str += i
        # content = re.sub("&nbsp;&nbsp;", '', content_str)
        # content = re.sub("<br />", '', content)
        # content = re.sub("</?font[^><]*>", '', content)
        # content = re.sub("</?a[^><]*>", "", content)
        # content = re.sub("|'\'[]];'./<>?,/!@#$%^&*()-=_+`", '', content)
        # content = content.replace(" ", "")
        item['content']=strings


        yield item
