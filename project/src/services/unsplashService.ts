// Unsplash URL generation service for country-specific background images
// Uses Unsplash Source API which doesn't require API keys

// Generate deterministic Unsplash URLs based on country search terms
export const generateUnsplashUrl = (searchTerms: string, width = 1200, height = 800): string => {
  // Use Unsplash Source API to get images by search term
  // This approach is more reliable and doesn't require API keys
  const encodedSearch = encodeURIComponent(searchTerms);
  return `https://source.unsplash.com/${width}x${height}/?${encodedSearch}`;
};

// Country-specific search terms optimized for beautiful background images
const getCountrySearchTerms = (countryName: string): string => {
  const terms: Record<string, string> = {
    // Major countries with iconic landmarks
    'japan': 'japan,mount-fuji,cherry-blossom,temple',
    'korea': 'south-korea,seoul,traditional-architecture,landscape',
    'france': 'france,paris,eiffel-tower,architecture',
    'italy': 'italy,venice,architecture,landscape',
    'germany': 'germany,castle,bavaria,architecture',
    'uk': 'england,london,architecture,countryside',
    'usa': 'america,new-york,landscape,landmarks',
    'china': 'china,great-wall,architecture,landscape',
    'thailand': 'thailand,temple,bangkok,tropical',
    'vietnam': 'vietnam,ha-long-bay,landscape,temple',
    'brazil': 'brazil,rio-de-janeiro,christ-redeemer,landscape',
    'india': 'india,taj-mahal,architecture,palace',
    'australia': 'australia,sydney-opera-house,landscape,architecture',
    'canada': 'canada,rockies,nature,landscape',
    'spain': 'spain,sagrada-familia,architecture,landscape',
    'netherlands': 'netherlands,amsterdam,canal,windmill',
    'finland': 'finland,northern-lights,aurora,nature',
    'iceland': 'iceland,blue-lagoon,nature,landscape',
    'norway': 'norway,fjord,northern-lights,nature',
    'sweden': 'sweden,stockholm,nature,architecture',
    'denmark': 'denmark,copenhagen,architecture,landscape',
    'switzerland': 'switzerland,alps,matterhorn,mountain',
    'austria': 'austria,salzburg,alps,architecture',
    'portugal': 'portugal,lisbon,architecture,landscape',
    'greece': 'greece,santorini,islands,architecture',
    'turkey': 'turkey,istanbul,architecture,landscape',
    'egypt': 'egypt,pyramids,giza,desert',
    'morocco': 'morocco,marrakech,architecture,desert',
    'russia': 'russia,moscow,red-square,architecture',
    'mexico': 'mexico,chichen-itza,architecture,landscape',
    'argentina': 'argentina,buenos-aires,patagonia,landscape',
    'chile': 'chile,atacama-desert,andes,landscape',
    'peru': 'peru,machu-picchu,mountains,ancient',
    'new zealand': 'new-zealand,milford-sound,nature,landscape',
    'philippines': 'philippines,palawan,islands,tropical',
    'indonesia': 'indonesia,bali,temple,tropical',
    'malaysia': 'malaysia,kuala-lumpur,petronas-towers,architecture',
    'singapore': 'singapore,marina-bay,skyline,architecture',
    'south africa': 'south-africa,cape-town,table-mountain,landscape',
    'algeria': 'algeria,sahara,desert,landscape',
    'bhutan': 'bhutan,monastery,mountains,architecture',
    'maldives': 'maldives,beach,tropical,paradise',
    'malta': 'malta,valletta,architecture,mediterranean',
    'luxembourg': 'luxembourg,castle,architecture,landscape',
    'brunei': 'brunei,mosque,architecture,islamic',
    'montenegro': 'montenegro,kotor,bay,landscape',
    'estonia': 'estonia,tallinn,medieval,architecture',
    'latvia': 'latvia,riga,architecture,baltic',
    'lithuania': 'lithuania,vilnius,architecture,baltic',
    'slovenia': 'slovenia,bled,lake,landscape',
    'cyprus': 'cyprus,coast,mediterranean,landscape',
    'andorra': 'andorra,mountains,pyrenees,landscape',
    'san-marino': 'san-marino,towers,architecture,medieval',
    'monaco': 'monaco,monte-carlo,luxury,architecture',
    'vatican': 'vatican,st-peters,architecture,basilica',
    'seychelles': 'seychelles,beach,tropical,paradise',
    'palau': 'palau,jellyfish-lake,tropical,diving',
    'tuvalu': 'tuvalu,atoll,pacific,tropical',
    'nauru': 'nauru,island,pacific,tropical',
    'kiribati': 'kiribati,atoll,pacific,tropical',
    'marshall-islands': 'marshall-islands,atoll,pacific,tropical',
    'micronesia': 'micronesia,diving,pacific,tropical',
    'tonga': 'tonga,pacific,tropical,islands',
    'samoa': 'samoa,pacific,tropical,beach',
    'vanuatu': 'vanuatu,volcano,pacific,tropical',
    'fiji': 'fiji,tropical,beach,paradise',
    // Add more countries as needed
    'afghanistan': 'afghanistan,mountains,landscape,architecture',
    'albania': 'albania,riviera,mountains,landscape',
    'armenia': 'armenia,monastery,mountains,architecture',
    'azerbaijan': 'azerbaijan,baku,caspian,architecture',
    'bahrain': 'bahrain,manama,architecture,gulf',
    'bangladesh': 'bangladesh,dhaka,landscape,architecture',
    'belarus': 'belarus,minsk,architecture,landscape',
    'belgium': 'belgium,brussels,architecture,medieval',
    'bosnia': 'bosnia,sarajevo,bridge,architecture',
    'bulgaria': 'bulgaria,sofia,mountains,architecture',
    'croatia': 'croatia,dubrovnik,coast,architecture',
    'czech': 'czech-republic,prague,architecture,castle',
    'hungary': 'hungary,budapest,danube,architecture',
    'iran': 'iran,isfahan,mosque,architecture',
    'iraq': 'iraq,baghdad,mesopotamia,architecture',
    'israel': 'israel,jerusalem,architecture,landscape',
    'jordan': 'jordan,petra,desert,ancient',
    'kazakhstan': 'kazakhstan,almaty,mountains,landscape',
    'kuwait': 'kuwait,towers,architecture,gulf',
    'kyrgyzstan': 'kyrgyzstan,mountains,landscape,nature',
    'lebanon': 'lebanon,beirut,cedars,mountains',
    'moldova': 'moldova,chisinau,architecture,landscape',
    'mongolia': 'mongolia,steppe,landscape,nature',
    'myanmar': 'myanmar,bagan,temple,pagoda',
    'nepal': 'nepal,everest,himalayas,temple',
    'oman': 'oman,muscat,architecture,desert',
    'pakistan': 'pakistan,k2,mountains,landscape',
    'qatar': 'qatar,doha,skyline,architecture',
    'saudi arabia': 'saudi-arabia,riyadh,desert,architecture',
    'serbia': 'serbia,belgrade,architecture,landscape',
    'slovakia': 'slovakia,bratislava,castle,architecture',
    'sri lanka': 'sri-lanka,temple,tea-plantation,landscape',
    'syria': 'syria,damascus,architecture,ancient',
    'tajikistan': 'tajikistan,mountains,pamir,landscape',
    'turkmenistan': 'turkmenistan,ashgabat,desert,architecture',
    'uae': 'uae,dubai,burj-khalifa,architecture',
    'ukraine': 'ukraine,kiev,architecture,landscape',
    'uzbekistan': 'uzbekistan,samarkand,architecture,islamic',
    'yemen': 'yemen,sanaa,architecture,mountains'
  };

  // Return specific terms or fallback to generic country search
  return terms[countryName.toLowerCase()] || `${countryName},landscape,nature,architecture`;
};

// Main function to get country-specific Unsplash image URL
export const getUnsplashCountryImage = (countryName: string): string => {
  const searchTerms = getCountrySearchTerms(countryName);
  return generateUnsplashUrl(searchTerms);
};

// Alternative function that provides multiple variations
export const getUnsplashCountryImageVariations = (countryName: string): string[] => {
  const baseTerms = getCountrySearchTerms(countryName);
  const variations = [
    baseTerms,
    `${countryName},architecture,landmark`,
    `${countryName},nature,landscape`,
    `${countryName},city,skyline`,
    `${countryName},culture,traditional`
  ];

  return variations.map(terms => generateUnsplashUrl(terms));
};

// Function to get image with fallback
export const getCountryImageWithFallback = (countryName: string): string => {
  try {
    return getUnsplashCountryImage(countryName);
  } catch (error) {
    console.warn(`Failed to generate Unsplash URL for ${countryName}:`, error);
    // Fallback to existing image
    return 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&h=800&fit=crop&q=80';
  }
};