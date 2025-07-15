"""
画像URL自動取得システム
Unsplash APIを使用して各国の代表的な画像URLを取得
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from schemas.country_schema import CountryImages


class ImageUrlGenerator:
    """画像URL自動生成クラス"""
    
    def __init__(self, unsplash_access_key: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        
        # Unsplash API設定
        self.unsplash_access_key = unsplash_access_key
        self.unsplash_base_url = "https://api.unsplash.com"
        
        # 既存の画像マッピングを読み込み
        self.existing_images = self._load_existing_images()
        
        # 国別検索キーワード定義
        self.country_search_keywords = self._define_country_search_keywords()
        
        # フォールバック画像URL
        self.fallback_image = "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop&q=80"
    
    def _load_existing_images(self) -> Dict[str, str]:
        """既存の画像マッピングを読み込み"""
        try:
            # フロントエンドのcountryImages.tsから既存のマッピングを抽出
            frontend_images_path = self.base_dir.parent / "project" / "src" / "data" / "countryImages.ts"
            existing_images = {}
            
            if frontend_images_path.exists():
                with open(frontend_images_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 簡易的な正規表現でURLを抽出
                import re
                matches = re.findall(r"'([^']+)':\s*'(https://[^']+)'", content)
                for country_code, url in matches:
                    existing_images[country_code] = url
            
            return existing_images
        except Exception as e:
            print(f"Warning: Could not load existing images: {e}")
            return {}
    
    def _define_country_search_keywords(self) -> Dict[str, List[str]]:
        """国別検索キーワードを定義"""
        return {
            # アジア
            "japan": ["japan", "tokyo", "kyoto", "mount fuji", "japanese temple"],
            "korea": ["south korea", "seoul", "korean palace", "busan", "jeju"],
            "china": ["china", "great wall", "beijing", "shanghai", "chinese architecture"],
            "india": ["india", "taj mahal", "mumbai", "rajasthan", "indian temple"],
            "thailand": ["thailand", "bangkok", "thai temple", "phuket", "chiang mai"],
            "vietnam": ["vietnam", "hanoi", "ho chi minh", "halong bay", "vietnamese"],
            "singapore": ["singapore", "marina bay", "singapore skyline", "gardens by the bay"],
            "malaysia": ["malaysia", "kuala lumpur", "petronas towers", "langkawi"],
            "indonesia": ["indonesia", "bali", "jakarta", "borobudur", "indonesian"],
            "philippines": ["philippines", "manila", "boracay", "palawan", "filipino"],
            "cambodia": ["cambodia", "angkor wat", "phnom penh", "cambodian temple"],
            "myanmar": ["myanmar", "yangon", "mandalay", "bagan", "myanmar temple"],
            "laos": ["laos", "vientiane", "luang prabang", "mekong", "laotian"],
            "brunei": ["brunei", "bandar seri begawan", "sultan omar mosque"],
            "mongolia": ["mongolia", "ulaanbaatar", "mongolian steppe", "ger"],
            "bhutan": ["bhutan", "thimphu", "tigers nest", "bhutanese monastery"],
            "nepal": ["nepal", "kathmandu", "mount everest", "himalaya", "nepali"],
            "sri lanka": ["sri lanka", "colombo", "sigiriya", "kandy", "sri lankan"],
            "bangladesh": ["bangladesh", "dhaka", "cox's bazar", "bengali"],
            "maldives": ["maldives", "maldivian", "tropical beach", "overwater villa"],
            "afghanistan": ["afghanistan", "kabul", "afghan landscape", "hindu kush"],
            "pakistan": ["pakistan", "islamabad", "lahore", "karachi", "pakistani"],
            
            # 欧州
            "united kingdom": ["london", "big ben", "england", "scotland", "wales"],
            "france": ["paris", "eiffel tower", "versailles", "french", "provence"],
            "germany": ["germany", "berlin", "neuschwanstein", "bavarian", "german"],
            "italy": ["italy", "rome", "venice", "florence", "italian"],
            "spain": ["spain", "madrid", "barcelona", "sagrada familia", "spanish"],
            "russia": ["russia", "moscow", "st petersburg", "red square", "russian"],
            "netherlands": ["netherlands", "amsterdam", "dutch", "windmill", "tulip"],
            "belgium": ["belgium", "brussels", "bruges", "belgian", "atomium"],
            "switzerland": ["switzerland", "zurich", "matterhorn", "swiss alps", "swiss"],
            "austria": ["austria", "vienna", "salzburg", "austrian alps", "austrian"],
            "sweden": ["sweden", "stockholm", "swedish", "scandinavian", "northern lights"],
            "norway": ["norway", "oslo", "fjord", "northern lights", "norwegian"],
            "denmark": ["denmark", "copenhagen", "danish", "scandinavian", "nyhavn"],
            "finland": ["finland", "helsinki", "lapland", "northern lights", "finnish"],
            "iceland": ["iceland", "reykjavik", "blue lagoon", "northern lights", "icelandic"],
            "ireland": ["ireland", "dublin", "irish", "cliffs of moher", "emerald isle"],
            "portugal": ["portugal", "lisbon", "porto", "portuguese", "azores"],
            "greece": ["greece", "athens", "santorini", "greek", "acropolis"],
            "turkey": ["turkey", "istanbul", "cappadocia", "turkish", "hagia sophia"],
            "poland": ["poland", "warsaw", "krakow", "polish", "zakopane"],
            "czech republic": ["czech republic", "prague", "bohemian", "czech"],
            "hungary": ["hungary", "budapest", "hungarian", "parliament building"],
            "romania": ["romania", "bucharest", "transylvania", "romanian", "carpathian"],
            "bulgaria": ["bulgaria", "sofia", "plovdiv", "bulgarian", "rila monastery"],
            "croatia": ["croatia", "zagreb", "dubrovnik", "plitvice", "croatian"],
            "serbia": ["serbia", "belgrade", "novi sad", "serbian", "orthodox"],
            "bosnia": ["bosnia", "sarajevo", "mostar", "bosnian", "ottoman"],
            "albania": ["albania", "tirana", "albanian", "mediterranean", "balkans"],
            "montenegro": ["montenegro", "podgorica", "kotor", "montenegrin", "adriatic"],
            "slovenia": ["slovenia", "ljubljana", "bled", "slovenian", "alps"],
            "slovakia": ["slovakia", "bratislava", "slovak", "high tatras", "spiš"],
            "estonia": ["estonia", "tallinn", "estonian", "baltic", "medieval"],
            "latvia": ["latvia", "riga", "latvian", "baltic", "art nouveau"],
            "lithuania": ["lithuania", "vilnius", "lithuanian", "baltic", "trakai"],
            "belarus": ["belarus", "minsk", "belarusian", "mir castle", "nesvizh"],
            "ukraine": ["ukraine", "kyiv", "lviv", "ukrainian", "carpathian"],
            "moldova": ["moldova", "chisinau", "moldovan", "orheiul vechi"],
            "georgia": ["georgia", "tbilisi", "georgian", "caucasus", "svaneti"],
            "armenia": ["armenia", "yerevan", "armenian", "ararat", "geghard"],
            "azerbaijan": ["azerbaijan", "baku", "azerbaijani", "caspian", "caucasus"],
            
            # 北米
            "united states": ["usa", "new york", "grand canyon", "statue of liberty"],
            "canada": ["canada", "toronto", "vancouver", "banff", "canadian"],
            "mexico": ["mexico", "mexico city", "cancun", "chichen itza", "mexican"],
            
            # 南米
            "brazil": ["brazil", "rio de janeiro", "christ redeemer", "amazon", "brazilian"],
            "argentina": ["argentina", "buenos aires", "patagonia", "tango", "argentinian"],
            "chile": ["chile", "santiago", "atacama", "torres del paine", "chilean"],
            "peru": ["peru", "lima", "machu picchu", "cusco", "peruvian"],
            "colombia": ["colombia", "bogota", "cartagena", "colombian", "coffee"],
            "venezuela": ["venezuela", "caracas", "angel falls", "venezuelan", "tepui"],
            "ecuador": ["ecuador", "quito", "galapagos", "ecuadorian", "andes"],
            "bolivia": ["bolivia", "la paz", "salar de uyuni", "bolivian", "altiplano"],
            "paraguay": ["paraguay", "asuncion", "paraguayan", "guarani", "pantanal"],
            "uruguay": ["uruguay", "montevideo", "uruguayan", "punta del este"],
            "guyana": ["guyana", "georgetown", "guyanese", "kaieteur falls"],
            "suriname": ["suriname", "paramaribo", "surinamese", "rainforest"],
            
            # アフリカ
            "south africa": ["south africa", "cape town", "kruger", "table mountain"],
            "egypt": ["egypt", "cairo", "pyramids", "sphinx", "nile"],
            "morocco": ["morocco", "casablanca", "marrakech", "sahara", "moroccan"],
            "algeria": ["algeria", "algiers", "sahara", "tassili", "algerian"],
            "tunisia": ["tunisia", "tunis", "carthage", "sahara", "tunisian"],
            "libya": ["libya", "tripoli", "libyan", "sahara", "leptis magna"],
            "sudan": ["sudan", "khartoum", "sudanese", "nile", "nubian"],
            "ethiopia": ["ethiopia", "addis ababa", "lalibela", "ethiopian", "danakil"],
            "kenya": ["kenya", "nairobi", "masai mara", "mount kenya", "kenyan"],
            "tanzania": ["tanzania", "dar es salaam", "kilimanjaro", "serengeti"],
            "uganda": ["uganda", "kampala", "ugandan", "gorilla", "nile"],
            "rwanda": ["rwanda", "kigali", "rwandan", "gorilla", "virunga"],
            "nigeria": ["nigeria", "lagos", "abuja", "nigerian", "yoruba"],
            "ghana": ["ghana", "accra", "kumasi", "ghanaian", "ashanti"],
            "senegal": ["senegal", "dakar", "senegalese", "goree island"],
            "mali": ["mali", "bamako", "timbuktu", "malian", "dogon"],
            "burkina faso": ["burkina faso", "ouagadougou", "burkinabe", "mossi"],
            "niger": ["niger", "niamey", "nigerien", "sahara", "tuareg"],
            "chad": ["chad", "ndjamena", "chadian", "sahara", "tibesti"],
            "cameroon": ["cameroon", "yaounde", "douala", "cameroonian", "mount cameroon"],
            "central african republic": ["central african republic", "bangui", "sangha"],
            "congo": ["congo", "brazzaville", "congolese", "congo river"],
            "democratic republic of congo": ["congo", "kinshasa", "virunga", "congo river"],
            "gabon": ["gabon", "libreville", "gabonese", "loango", "rainforest"],
            "equatorial guinea": ["equatorial guinea", "malabo", "bioko", "rainforest"],
            "sao tome and principe": ["sao tome", "sao tome and principe", "tropical"],
            "namibia": ["namibia", "windhoek", "namib desert", "sossusvlei"],
            "botswana": ["botswana", "gaborone", "okavango", "kalahari"],
            "zimbabwe": ["zimbabwe", "harare", "victoria falls", "zimbabwean"],
            "zambia": ["zambia", "lusaka", "victoria falls", "zambian"],
            "malawi": ["malawi", "lilongwe", "lake malawi", "malawian"],
            "mozambique": ["mozambique", "maputo", "mozambican", "indian ocean"],
            "madagascar": ["madagascar", "antananarivo", "baobab", "malagasy"],
            "mauritius": ["mauritius", "port louis", "mauritian", "tropical beach"],
            "seychelles": ["seychelles", "victoria", "seychellois", "tropical beach"],
            "comoros": ["comoros", "moroni", "comorian", "indian ocean"],
            "cape verde": ["cape verde", "praia", "cabo verde", "atlantic"],
            "guinea": ["guinea", "conakry", "guinean", "fouta djallon"],
            "guinea bissau": ["guinea bissau", "bissau", "bijagos", "mangrove"],
            "sierra leone": ["sierra leone", "freetown", "sierra leonean", "atlantic"],
            "liberia": ["liberia", "monrovia", "liberian", "atlantic", "sapo"],
            "ivory coast": ["ivory coast", "abidjan", "yamoussoukro", "ivorian"],
            "togo": ["togo", "lome", "togolese", "koutammakou"],
            "benin": ["benin", "porto novo", "cotonou", "beninese", "pendjari"],
            "djibouti": ["djibouti", "djibouti city", "djiboutian", "lac assal"],
            "eritrea": ["eritrea", "asmara", "eritrean", "red sea"],
            "somalia": ["somalia", "mogadishu", "somali", "indian ocean"],
            "south sudan": ["south sudan", "juba", "south sudanese", "nile"],
            "angola": ["angola", "luanda", "angolan", "atlantic", "iona"],
            "lesotho": ["lesotho", "maseru", "lesotho", "drakensberg"],
            "swaziland": ["swaziland", "mbabane", "swazi", "hlane"],
            
            # オセアニア
            "australia": ["australia", "sydney", "melbourne", "uluru", "great barrier reef"],
            "new zealand": ["new zealand", "auckland", "wellington", "milford sound"],
            "fiji": ["fiji", "suva", "fijian", "tropical beach", "coral reef"],
            "papua new guinea": ["papua new guinea", "port moresby", "sepik", "highlands"],
            "solomon islands": ["solomon islands", "honiara", "solomon", "coral reef"],
            "vanuatu": ["vanuatu", "port vila", "vanuatuan", "volcano", "tropical"],
            "samoa": ["samoa", "apia", "samoan", "tropical beach", "polynesian"],
            "tonga": ["tonga", "nukualofa", "tongan", "tropical beach", "polynesian"],
            "kiribati": ["kiribati", "tarawa", "kiribati", "coral atoll", "pacific"],
            "tuvalu": ["tuvalu", "funafuti", "tuvaluan", "coral atoll", "pacific"],
            "nauru": ["nauru", "yaren", "nauruan", "phosphate", "pacific"],
            "palau": ["palau", "ngerulmud", "palauan", "jellyfish lake", "coral reef"],
            "marshall islands": ["marshall islands", "majuro", "marshallese", "coral atoll"],
            "micronesia": ["micronesia", "palikir", "micronesian", "coral reef", "pacific"],
            
            # 中東
            "israel": ["israel", "jerusalem", "tel aviv", "dead sea", "israeli"],
            "palestine": ["palestine", "ramallah", "gaza", "palestinian", "bethlehem"],
            "jordan": ["jordan", "amman", "petra", "wadi rum", "jordanian"],
            "lebanon": ["lebanon", "beirut", "baalbek", "cedar", "lebanese"],
            "syria": ["syria", "damascus", "aleppo", "palmyra", "syrian"],
            "iraq": ["iraq", "baghdad", "babylon", "mesopotamia", "iraqi"],
            "iran": ["iran", "tehran", "isfahan", "persepolis", "iranian"],
            "saudi arabia": ["saudi arabia", "riyadh", "mecca", "medina", "saudi"],
            "yemen": ["yemen", "sanaa", "socotra", "yemeni", "arabian"],
            "oman": ["oman", "muscat", "omani", "arabian", "nizwa"],
            "united arab emirates": ["uae", "dubai", "abu dhabi", "burj khalifa"],
            "qatar": ["qatar", "doha", "qatari", "arabian", "pearl"],
            "bahrain": ["bahrain", "manama", "bahraini", "arabian", "pearl"],
            "kuwait": ["kuwait", "kuwait city", "kuwaiti", "arabian", "towers"],
            
            # 中央アジア
            "kazakhstan": ["kazakhstan", "almaty", "nur sultan", "kazakh", "steppe"],
            "uzbekistan": ["uzbekistan", "tashkent", "samarkand", "bukhara", "uzbek"],
            "turkmenistan": ["turkmenistan", "ashgabat", "turkmen", "karakum", "darvaza"],
            "kyrgyzstan": ["kyrgyzstan", "bishkek", "kyrgyz", "tian shan", "issyk kul"],
            "tajikistan": ["tajikistan", "dushanbe", "tajik", "pamir", "fann mountains"],
        }
    
    def _search_unsplash_image(self, query: str, per_page: int = 1) -> Optional[str]:
        """Unsplash APIで画像を検索"""
        if not self.unsplash_access_key:
            print("Warning: Unsplash API key not provided, using fallback")
            return None
            
        try:
            url = f"{self.unsplash_base_url}/search/photos"
            params = {
                'query': query,
                'per_page': per_page,
                'orientation': 'landscape',
                'client_id': self.unsplash_access_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    photo = results[0]
                    # 適切なサイズのURLを生成
                    raw_url = photo['urls']['raw']
                    sized_url = f"{raw_url}&w=1200&h=800&fit=crop&q=80"
                    return sized_url
            else:
                print(f"Unsplash API error: {response.status_code}")
                
        except Exception as e:
            print(f"Error searching Unsplash: {e}")
            
        return None
    
    def _get_curated_image_url(self, country_code: str, country_name: str) -> str:
        """キュレートされた画像URLを取得"""
        # 既存の画像がある場合はそれを使用
        if country_code in self.existing_images:
            return self.existing_images[country_code]
        
        # 検索キーワードを取得
        keywords = self.country_search_keywords.get(country_name.lower(), [country_name])
        
        # 各キーワードで検索を試行
        for keyword in keywords:
            image_url = self._search_unsplash_image(keyword)
            if image_url:
                return image_url
        
        # 見つからない場合はフォールバック
        return self.fallback_image
    
    def generate_country_images(self, country_raw_data: Dict) -> Dict[str, CountryImages]:
        """国別の画像URLを生成"""
        country_images = {}
        
        for country_code, data in country_raw_data.items():
            country_name = data.get('nameEn', '')
            print(f"Generating images for {country_name} ({country_code})...")
            
            # プライマリ画像を取得
            primary_image = self._get_curated_image_url(country_code, country_name)
            
            # ハイライト画像（とりあえずプライマリと同じを使用）
            highlight_images = [primary_image] * 4
            
            country_images[country_code] = CountryImages(
                primary=primary_image,
                highlights=highlight_images,
                fallback=self.fallback_image
            )
            
            print(f"  ✓ Generated images for {country_name}")
            
            # APIレート制限対応
            if self.unsplash_access_key:
                time.sleep(1)
        
        return country_images
    
    def save_images_data(self, country_images: Dict[str, CountryImages], filename: str = "country_images.json"):
        """画像データをJSONファイルに保存"""
        output_file = self.data_dir / filename
        
        # CountryImagesオブジェクトを辞書に変換
        images_dict = {}
        for country_code, images in country_images.items():
            images_dict[country_code] = {
                "primary": images.primary,
                "highlights": images.highlights,
                "fallback": images.fallback
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(images_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Images data saved to {output_file}")
    
    def generate_typescript_images_file(self, country_images: Dict[str, CountryImages], 
                                       filename: str = "countryImages.ts"):
        """TypeScript形式の画像ファイルを生成"""
        output_file = self.base_dir.parent / "project" / "src" / "data" / filename
        
        # TypeScriptファイルの内容を生成
        content = """// 各国の厳選された美しい画像URL（自動生成）
export const countryImages: Record<string, string> = {
"""
        
        for country_code, images in country_images.items():
            content += f'  "{country_code}": "{images.primary}",\n'
        
        content += """};

// フォールバック画像
export const getFallbackImage = (countryName: string): string => {
  return "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200&h=800&fit=crop&q=80";
};

// 画像取得関数
export const getCountryImage = (countryCode: string, countryName: string): string => {
  return countryImages[countryCode.toLowerCase()] || getFallbackImage(countryName);
};
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ TypeScript images file generated: {output_file}")
    
    def run_with_raw_data(self, country_raw_data: Dict, unsplash_key: Optional[str] = None):
        """生データを使用して画像生成を実行"""
        if unsplash_key:
            self.unsplash_access_key = unsplash_key
        
        print("🖼️  Image URL Generator")
        print("=" * 50)
        
        # 画像URLを生成
        print("🔍 Generating image URLs...")
        country_images = self.generate_country_images(country_raw_data)
        
        # JSONファイルに保存
        print("\n💾 Saving images data...")
        self.save_images_data(country_images)
        
        # TypeScriptファイルを生成
        print("\n📝 Generating TypeScript images file...")
        self.generate_typescript_images_file(country_images)
        
        print("\n✅ Image URL generation completed!")
        print(f"   Generated images for: {len(country_images)} countries")
        
        return country_images


if __name__ == "__main__":
    # 単体テスト用
    generator = ImageUrlGenerator()
    
    # テスト用の生データ
    test_data = {
        "jp": {"nameEn": "Japan"},
        "us": {"nameEn": "United States"},
        "fr": {"nameEn": "France"}
    }
    
    generator.run_with_raw_data(test_data)