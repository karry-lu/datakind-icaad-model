#hard-code a dictionary containing regex's that trigger SDGs
#key = 'SDG_#'; value = list of regex's for that SDG

def load_keyword_dict(): 
	keyword_dict = dict()

	keyword_dict['SDG_1'] = [
	'(?=poverty)(?=eradic|alleviat|reduc|eliminat|fight|combat)',
	'poverty line',
	'extreme poverty',
	'socioeconomic development',
	'(?=sufficient)(?=standard of living)',
	'poorest sectors of the population',
	'access to (economic resources|credit|microcredit|land)',
	'(?=equity|equal)(?=wealth distribution)',
	'(?=reduc)(?=economic disparit)',
	'low(.*)income population',
	'rich and the poor',
	'poor and the rich'
	]

	keyword_dict['SDG_2'] = [
	'food (security|insecurity|provision|supply|distribut|self(.*)sufficiency|assistance|produc)',
	'(right to|transfer of|access to|generation of self(.*)consumption) food',
	'combat hunger',
	'(fight against mal|right to) nutrition',
	'agricultural system of higher productivity',
	'sustainable agricultural practice'
	]

	keyword_dict['SDG_3'] = [
	'(maternal|child|infant) morality',
	'low life expectancy',
	'high mortality rate',
	'(right to|area of|field of|promot|maternal) health',
	'health (policy|need|reform)',
	'(?=disabilit)(?=needs)',
	'combat (hiv|aids)',
	'vaccin',
	'medical facilit',
	'(sexual|reproductive) (right|health|product)',
	'safe drinking water',
	'wastewater',
	'health infrastructure',
	'(?=develop)(?=health sector)',
	'public health',
	'personal hygiene',
	'health education',
	'(?=basic services)(?=health)',
	'well(-| )fed'
	]

	keyword_dict['SDG_4'] = [
	'(access to|right to|improvement of|discrimination in|budget allocation for|disparities in|reinforce girls|strengthen public) education',
	'school (system|enrollment rate|environment)',
	'eliminate illiteracy',
	'educational (opportunit|institution|system)',
	'teaching',
	'education (of minorit)',
	'(?=teach)(?=minority language)',
	'(?=inclusive education)(?=disabilit)',
	'reduce the high dropout rate in(.+)school',
	'culture of human rights',
	'(free|compulsory|public|health) education',
	'human rights (education|training)',
	'training( programmes)* on human rights',
	'(?=basic services)(?=education)',
	'well(-| )educated'
	]

	keyword_dict['SDG_5'] = [
	'gender',
	#'gender (issue|perspective|equality|equity|inequality|strategy|discrimination|disparit|gap|imbalance)',
	'mothers',
	'women',
	'female',
	'pregnant',
	'nursing',
	'girl',
	'maternal (health|morality|morbidity)',
	'(sexual|reproductive) rights',
	'reproductive health (service|product)',
	'(sexual|domestic) violence',
	'single parent famil',
	'domestic worker',
	'sexual exploitation',
	'forced (marriage|prostitution)',
	'sex traffick',
	'(?=ilo convention)(?=189)',
	'equal (remuneration|pay)',
	'special marriage act',
	'sexual offen(c|s)e'
	]

	keyword_dict['SDG_6'] = [
	'(access to|right to|issue of) water',
	'(?=water)(?=sanitation)',
	'safe drinking water',
	'water (resource|coverage|law)',
	'basic sanitation',
	'sanitation (facilit|coverage)',
	'wastewater'
	]

	keyword_dict['SDG_7'] = [
	'industrial (infrastructure|pollution)',
	'(?=guarantee)(?=better service)',
	'access to basic (service|infrastructure)',
	'power stations',
	'energy',
	'greenhouse gas emission',
	'(pollutant|low impact) technology ',
	'(?=pollution)(?=emission)',
	'green economy',
	'air quality'
	]

	keyword_dict['SDG_8'] = [
	'(socioeconomic|tourism) development',
	'economic (potential|development|impact|wellbeing|growth)',
	'(?=equity)(?=wealth distribution)',
	'underdevelopment',
	'extractive industries transparency initiative',
	'(right to|disparities in|standards of|rural) employment',
	'promotion of work',
	'employment of youth',
	'(fight against|reduce) unemployment',
	'creating job',
	'(employment|job) opportunit',
	'(?=ilo convention)(?=2)',
	'ilo equal remuneration convention',
	'(?=support for)(?=enterprise)',
	'income generating strateg',
	'(?=safety)(?=workplace)',
	'(?=discrimination|violence)(?=employment)'
	'working condition',
	'(health of|rights of) workers',
	'worker rights',
	'labour (law|right|market|standard)',
	'trade union',
	'migrant worker',
	'icrmw',
	'(workplace|labour) inspection',
	'Migrant workers',
	'(child|forced) labour',
	'domestic workers',
	'right to (collective bargain|strike)',
	'occupational (health|safety)',
	'trafficker',
	'professional protection',
	'ilo equal remuneration convention',
	'business and human rights',
	'(?=ilo convention)(?=189)',
	'(?=ilo convention)(?=117)',
	'micro(credit|finance)',
	'(?=equal opportunit)(?=work)'
	]

	keyword_dict['SDG_9'] = [
	'infrastructure',
	'industrial (activit|development)',
	'economic development',
	'small-scale enterprise',
	'microcredit'
	]

	keyword_dict['SDG_10'] = [
	'ethnic',
	'racial',
	'minorit',
	'gender equality',
	'disabilit',
	'(tribal|indigenous) (person|people|population|communit)',
	'discrimination',
	'roma',
	'migration',
	'icrmw',
	'(?=ilo convention)(?=189)',
	'(migrant|domestic) worker',
	'marginalized',
	'disparit',
	'(reduce|bridge) the gap',
	'vulnerable (people|group|sectors of the population)',
	'urban(.+)rural inequality',
	'discriminatory (legislation|law|practice)',
	'excluded',
	'social inequality',
	'aboriginal people',
	'women',
	'girls',
	'exclusion',
	'regardless of ethnicity',
	'rural area',
	'immigrant',
	'equal (access|opportunit)',
	'promotion of equity',
	'inclusive education',
	'geographic inequalit',
	'(?=redistribut)(?=income)',
	'disadvantaged',
	'social inclusion strategy',
	'equality in income',
	'special marriage act'
	]

	keyword_dict['SDG_11'] = [
	'housing',
	'well(-| )housed'
	'eviction',
	'land (ownership|issue|allocation|concession|restitution)',
	'involuntary relocation',
	'forced dispossession',
	'rights to land',
	'forcibly removed from their land',
	'displacement',
	'(?=improve)(?=shelter)',
	'better service',
	'basic service',
	'construction',
	'integrated municipal-rural development',
	'(?=teaching)(?=minority language)',
	'disaster risk management',
	'(?=risk|mitigat)(?=natural disaster)',
	'greenhouse gas emissions',
	'pollution',
	'climate change',
	'air quality',
	'rural (area|communit|infrastructure|growth)',
	'remote (area|communit)',
	'(urban-rural|rural-urban) (inequalit|divide)'
	]

	keyword_dict['SDG_12'] = [
	'natural resource',
	'management of resource',
	'sustainable environment',
	'environmental (sustainability|impact|right)',
	'(?=extract)(?=oil compan)',
	'mining',
	'oil',
	'extractive ',
	'logging',
	'diamond',
	'principles on business and human right',
	'pollut',
	'emission',
	'toxic waste',
	'air quality',
	'(?=health hazard)(?=industrial activit)',
	'(?=corporate|private sector)(?=social responsibilit)',
	'(?=corporate sector)(?=social development)',
	'(?=companies)(?=accountab)',
	'(?=human rights perspective)(?=business)',
	'(trans|multi)national corporation',
	'(?=private sector)(?=accountab)',
	'(transparent|responsible) governance',
	'industrial compan',
	'(?=harmonious development)(?=people|nature)'
	]

	keyword_dict['SDG_13'] = [
	'environmental (develop|protect|right|service)',
	'approach to the environment',
	'(safe|healthy) environment',
	'climate (change|challenge)',
	'natural disaster',
	'disaster management',
	'greenhouse gas emission',
	'global warming'
	]

	keyword_dict['SDG_14'] = [
	'contamination of rivers',
	'natural resource',
	'(right to|access to|generation of self(.+)consumption) food',
	'food (security|self(.+)sufficiency)',
	'environmental protection',
	'sustainable development'
	]

	keyword_dict['SDG_15'] = [
	'Environmental (conservation|protection|degradation)',
	'sustainable environment',
	'(conservation|sustainable use) of natural resources',
	'deforestation',
	'forest degradation',
	'logging industr'
	]

	keyword_dict['SDG_16'] = [
	'offen(c|s)es against minors',
	'(?=protect)(?=child)(?=right)',
	'child (labour|protection mechanism|worker)',
	'(violence against|exploitation of) child',
	'access to justice',
	'fair trial',
	'compensation',
	'reparation',
	'forced eviction',
	'victims and land Rrestitution law',
	'effective intervention and investigation',
	'judicial reform',
	'legal counsel',
	'just and transparent process',
	'judicial mechanisms',
	'(?=resolv)(?=conflict)',
	'(?=strengthen)(?=applicable legal framework)',
	'traffick',
	'forced prostitution',
	'good governance',
	'institutional capacit',
	'(?=accountab)(?=government)',
	'strengthen relevant institution',
	'(?=prison)(?=condition)',
	'right to (participation|be consulted)',
	'consultation process',
	'(?=indigenous)(?=decision)',
	'prior consultation',
	'dialogue and negotiation',
	'in consultation with (.+) people',
	'in compliance with',
	'in line with',
	'implement the recommendation of',
	'international standard',
	'ratify',
	'(?=mainstream)(?=international obligation)',
	'special rapporteur',
	'(?=repeal)(?=discriminatory legislation)',
	'(judicial|legal) officials',
	'implement treaty body recommendation',
	'national legislation',
	'death penalty',
	'execution',
	'capital punishment'
	]

	keyword_dict['SDG_17'] = [
	'international (communit|support)',
	'in partnership with',
	'share best practices',
	'civil society ',
	'bring oda up to the internationally committed 0.7 per cent of gdp',
	'(financial|technical) assistance',
	'Financial assistance',
	'(?=cooperat)(?=regional|international|united nations bodies)',
	'relevant stakeholder'
	]

	return keyword_dict