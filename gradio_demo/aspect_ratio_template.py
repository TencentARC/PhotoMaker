# Note: Since output width & height need to be divisible by 8, the w & h -values do
#       not exactly match the stated aspect ratios... but they are "close enough":)

aspect_ratio_list = [
    {
        "name": "Instagram (1:1)",
        "w": 1024,
        "h": 1024,
    },
    {
        "name": "35mm film / Landscape (3:2)",
        "w": 1024,
        "h": 680,
    },
    {
        "name": "35mm film / Portrait (2:3)",
        "w": 680,
        "h": 1024,
    },
    {
        "name": "CRT Monitor / Landscape (4:3)",
        "w": 1024,
        "h": 768,
    },
    {
        "name": "CRT Monitor / Portrait (3:4)",
        "w": 768,
        "h": 1024,
    },
    {
        "name": "Widescreen TV / Landscape (16:9)",
        "w": 1024,
        "h": 576,
    },
    {
        "name": "Widescreen TV / Portrait (9:16)",
        "w": 576,
        "h": 1024,
    },
    {
        "name": "Widescreen Monitor / Landscape (16:10)",
        "w": 1024,
        "h": 640,
    },
    {
        "name": "Widescreen Monitor / Portrait (10:16)",
        "w": 640,
        "h": 1024,
    },
    {
        "name": "Cinemascope (2.39:1)",
        "w": 1024,
        "h": 424,
    },
    {
        "name": "Widescreen Movie (1.85:1)",
        "w": 1024,
        "h": 552,
    },
    {
        "name": "Academy Movie (1.37:1)",
        "w": 1024,
        "h": 744,
    },
    {
        "name": "Sheet-print (A-series) / Landscape (297:210)",
        "w": 1024,
        "h": 720,
    },
    {
        "name": "Sheet-print (A-series) / Portrait (210:297)",
        "w": 720,
        "h": 1024,
    },
]

aspect_ratios = {k["name"]: (k["w"], k["h"]) for k in aspect_ratio_list}
