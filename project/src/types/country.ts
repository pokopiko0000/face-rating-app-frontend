export interface CountryHighlight {
  title: string;
  description: string;
}

export interface CountryBasicInfo {
  capital: string;
  population: string;
  language: string;
}

export interface CountryCoordinates {
  lat: number;
  lng: number;
}

export interface CountryData {
  name: string;
  nameEn: string;
  flag: string;
  code: string;
  basic: CountryBasicInfo;
  coordinates: CountryCoordinates;
  description: string;
  highlights: CountryHighlight[];
  whyVisit: string;
}

export type CountryDataMap = Record<string, CountryData>;