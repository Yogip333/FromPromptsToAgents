"""
Recipe MCP Server - Food Analysis Tools
========================================
Session 5: The Challenge - Robotic Chef Platform

This MCP server provides tools for analysing dishes, cooking techniques,
equipment specifications, and safety requirements. It contains a built-in
database of dishes with realistic culinary data.

Originally developed in Session 4 (Recipe Agent), included here as a
self-contained copy so Session 5 works independently.
"""

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Recipe Agent")

# ---------------------------------------------------------------------------
# Inline Dish Database
# ---------------------------------------------------------------------------

DISHES = {
    "pasta carbonara": {
        "name": "Pasta Carbonara",
        "cuisine": "Italian",
        "difficulty": "intermediate",
        "prep_time_minutes": 10,
        "cook_time_minutes": 20,
        "servings": 4,
        "ingredients": [
            {"item": "spaghetti", "quantity": "400g", "prep": "none"},
            {"item": "guanciale", "quantity": "200g", "prep": "cut into strips"},
            {"item": "egg yolks", "quantity": "6", "prep": "separated"},
            {"item": "whole eggs", "quantity": "2", "prep": "beaten"},
            {"item": "pecorino romano", "quantity": "100g", "prep": "finely grated"},
            {"item": "parmesan", "quantity": "50g", "prep": "finely grated"},
            {"item": "black pepper", "quantity": "to taste", "prep": "freshly ground"},
        ],
        "techniques": [
            {
                "name": "boiling",
                "description": "Cook pasta in salted boiling water until al dente",
                "precision": "medium",
                "temperature_c": 100,
                "duration_minutes": 10,
            },
            {
                "name": "rendering fat",
                "description": "Slowly render fat from guanciale over medium-low heat",
                "precision": "high",
                "temperature_c": 130,
                "duration_minutes": 8,
            },
            {
                "name": "emulsification",
                "description": "Combine egg mixture with hot pasta off heat to create creamy sauce without scrambling",
                "precision": "critical",
                "temperature_c": 65,
                "duration_minutes": 2,
            },
        ],
        "equipment": ["large pot", "sauté pan", "mixing bowl", "tongs", "colander"],
        "temperatures": {
            "water_boil": 100,
            "guanciale_render": 130,
            "sauce_emulsify": 65,
            "serving": 70,
        },
        "steps": [
            "Bring a large pot of salted water to a rolling boil.",
            "Cut guanciale into strips approximately 5mm wide.",
            "In a bowl, whisk egg yolks, whole eggs, and grated cheeses together.",
            "Cook guanciale in a dry sauté pan over medium-low heat until fat renders and edges crisp (8 min).",
            "Cook spaghetti in boiling water until al dente (about 10 min).",
            "Reserve 200ml pasta cooking water before draining.",
            "Remove sauté pan from heat. Add drained pasta to the guanciale pan.",
            "Pour egg and cheese mixture over pasta, tossing vigorously off heat.",
            "Add pasta water a splash at a time to achieve a silky, creamy consistency.",
            "Season generously with freshly ground black pepper and serve immediately.",
        ],
        "safety": [
            "Ensure water is at a full rolling boil before adding pasta.",
            "Render guanciale slowly to avoid grease splatter.",
            "Remove pan from heat before adding egg mixture to prevent scrambling.",
            "Serve immediately - carbonara does not reheat well.",
        ],
    },
    "souffle": {
        "name": "Cheese Soufflé",
        "cuisine": "French",
        "difficulty": "advanced",
        "prep_time_minutes": 20,
        "cook_time_minutes": 25,
        "servings": 4,
        "ingredients": [
            {"item": "butter", "quantity": "50g", "prep": "melted, plus extra for ramekins"},
            {"item": "plain flour", "quantity": "40g", "prep": "sifted"},
            {"item": "whole milk", "quantity": "300ml", "prep": "warmed"},
            {"item": "gruyère cheese", "quantity": "150g", "prep": "finely grated"},
            {"item": "egg yolks", "quantity": "4", "prep": "separated"},
            {"item": "egg whites", "quantity": "5", "prep": "at room temperature"},
            {"item": "dijon mustard", "quantity": "1 tsp", "prep": "none"},
            {"item": "salt and white pepper", "quantity": "to taste", "prep": "none"},
        ],
        "techniques": [
            {
                "name": "roux making",
                "description": "Cook butter and flour together to form a roux base",
                "precision": "high",
                "temperature_c": 120,
                "duration_minutes": 3,
            },
            {
                "name": "béchamel sauce",
                "description": "Gradually whisk warm milk into roux to create smooth sauce",
                "precision": "high",
                "temperature_c": 85,
                "duration_minutes": 5,
            },
            {
                "name": "whipping egg whites",
                "description": "Whip egg whites to stiff peaks with no trace of yolk or fat",
                "precision": "critical",
                "temperature_c": 20,
                "duration_minutes": 5,
            },
            {
                "name": "folding",
                "description": "Gently fold egg whites into cheese base to retain air",
                "precision": "critical",
                "temperature_c": 40,
                "duration_minutes": 2,
            },
            {
                "name": "baking",
                "description": "Bake in preheated oven without opening door",
                "precision": "critical",
                "temperature_c": 190,
                "duration_minutes": 25,
            },
        ],
        "equipment": [
            "soufflé ramekins",
            "saucepan",
            "whisk",
            "electric mixer",
            "spatula",
            "oven",
            "baking sheet",
        ],
        "temperatures": {
            "roux": 120,
            "bechamel": 85,
            "oven": 190,
            "egg_whites": 20,
        },
        "steps": [
            "Preheat oven to 190°C. Butter ramekins and dust with fine breadcrumbs or parmesan.",
            "Melt butter in a saucepan, stir in flour, and cook the roux for 2-3 minutes.",
            "Gradually whisk in warm milk to create a smooth béchamel. Cook until thickened.",
            "Remove from heat. Stir in grated gruyère, mustard, salt, and pepper.",
            "Beat in egg yolks one at a time.",
            "Whip egg whites to stiff peaks using an electric mixer.",
            "Fold one-third of egg whites into cheese base to loosen.",
            "Gently fold in remaining egg whites in two additions, preserving volume.",
            "Pour into prepared ramekins, filling to three-quarters.",
            "Run a thumb around the inner rim to create a 'top hat' rise.",
            "Place on a baking sheet and bake for 25 minutes. Do not open oven door.",
            "Serve immediately upon removal from oven.",
        ],
        "safety": [
            "Oven must be fully preheated before baking.",
            "Do not open oven door during baking or soufflé will collapse.",
            "Handle hot ramekins with oven gloves.",
            "Ensure no yolk contaminates egg whites or they will not whip.",
        ],
    },
    "sushi rolls": {
        "name": "Sushi Rolls (Maki)",
        "cuisine": "Japanese",
        "difficulty": "intermediate",
        "prep_time_minutes": 45,
        "cook_time_minutes": 20,
        "servings": 4,
        "ingredients": [
            {"item": "sushi rice", "quantity": "400g", "prep": "washed and drained"},
            {"item": "rice vinegar", "quantity": "60ml", "prep": "seasoned with sugar and salt"},
            {"item": "nori sheets", "quantity": "8", "prep": "halved if needed"},
            {"item": "fresh salmon", "quantity": "200g", "prep": "cut into thin strips"},
            {"item": "cucumber", "quantity": "1", "prep": "julienned"},
            {"item": "avocado", "quantity": "2", "prep": "thinly sliced"},
            {"item": "soy sauce", "quantity": "to serve", "prep": "none"},
            {"item": "wasabi", "quantity": "to serve", "prep": "none"},
            {"item": "pickled ginger", "quantity": "to serve", "prep": "none"},
        ],
        "techniques": [
            {
                "name": "rice cooking",
                "description": "Cook sushi rice with precise water ratio and rest period",
                "precision": "high",
                "temperature_c": 100,
                "duration_minutes": 15,
            },
            {
                "name": "rice seasoning",
                "description": "Season hot rice with vinegar mixture while fanning to cool",
                "precision": "high",
                "temperature_c": 60,
                "duration_minutes": 5,
            },
            {
                "name": "precision cutting",
                "description": "Cut fish into uniform strips with a sharp knife in single strokes",
                "precision": "critical",
                "temperature_c": 4,
                "duration_minutes": 10,
            },
            {
                "name": "rolling",
                "description": "Tightly roll sushi using bamboo mat with even pressure",
                "precision": "high",
                "temperature_c": 20,
                "duration_minutes": 3,
            },
        ],
        "equipment": [
            "rice cooker",
            "hangiri (wooden bowl)",
            "bamboo rolling mat",
            "sharp sushi knife",
            "cutting board",
            "fan",
        ],
        "temperatures": {
            "rice_cooking": 100,
            "rice_serving": 37,
            "fish_storage": 4,
            "room_temp_assembly": 20,
        },
        "steps": [
            "Wash sushi rice until water runs clear (5-6 rinses).",
            "Cook rice with 1:1.1 rice-to-water ratio. Rest for 10 min after cooking.",
            "Mix rice vinegar, sugar, and salt. Heat gently to dissolve.",
            "Transfer hot rice to a hangiri. Drizzle vinegar mixture while cutting and folding.",
            "Fan the rice while mixing to achieve a glossy finish. Cool to body temperature.",
            "Slice salmon into strips approximately 1cm wide by 8cm long.",
            "Place nori sheet shiny-side down on bamboo mat.",
            "Spread a thin, even layer of rice over nori, leaving a 1cm border at the top.",
            "Lay fish, cucumber, and avocado in a line across the centre.",
            "Roll tightly using the bamboo mat, applying even pressure.",
            "Wet a sharp knife. Slice each roll into 6-8 pieces with a single clean cut.",
            "Serve with soy sauce, wasabi, and pickled ginger.",
        ],
        "safety": [
            "Keep raw fish refrigerated at 4°C or below until ready to use.",
            "Use sushi-grade fish only from a reputable supplier.",
            "Keep hands wet with vinegar water when handling rice to prevent sticking.",
            "Clean cutting surfaces thoroughly after handling raw fish.",
        ],
    },
    "pizza margherita": {
        "name": "Pizza Margherita",
        "cuisine": "Italian",
        "difficulty": "intermediate",
        "prep_time_minutes": 120,
        "cook_time_minutes": 10,
        "servings": 4,
        "ingredients": [
            {"item": "tipo 00 flour", "quantity": "500g", "prep": "sifted"},
            {"item": "water", "quantity": "325ml", "prep": "lukewarm (35°C)"},
            {"item": "fresh yeast", "quantity": "3g", "prep": "crumbled"},
            {"item": "salt", "quantity": "10g", "prep": "fine"},
            {"item": "san marzano tomatoes", "quantity": "400g", "prep": "crushed by hand"},
            {"item": "fresh mozzarella", "quantity": "250g", "prep": "torn into pieces"},
            {"item": "fresh basil", "quantity": "handful", "prep": "leaves picked"},
            {"item": "extra virgin olive oil", "quantity": "2 tbsp", "prep": "none"},
        ],
        "techniques": [
            {
                "name": "dough kneading",
                "description": "Knead dough until smooth and elastic with strong gluten development",
                "precision": "high",
                "temperature_c": 25,
                "duration_minutes": 15,
            },
            {
                "name": "fermentation",
                "description": "Allow dough to ferment and rise at room temperature",
                "precision": "medium",
                "temperature_c": 25,
                "duration_minutes": 120,
            },
            {
                "name": "stretching",
                "description": "Stretch dough by hand into a thin round disc, avoiding rolling pin",
                "precision": "high",
                "temperature_c": 25,
                "duration_minutes": 3,
            },
            {
                "name": "high-heat baking",
                "description": "Bake pizza at maximum oven temperature on a preheated stone or steel",
                "precision": "high",
                "temperature_c": 300,
                "duration_minutes": 7,
            },
        ],
        "equipment": [
            "mixing bowl",
            "pizza stone or steel",
            "oven",
            "pizza peel",
            "bench scraper",
            "kitchen scale",
        ],
        "temperatures": {
            "water": 35,
            "fermentation": 25,
            "oven": 300,
            "serving": 75,
        },
        "steps": [
            "Dissolve yeast in lukewarm water (35°C). Let stand 5 minutes.",
            "Combine flour and salt in a large bowl. Add yeast water gradually.",
            "Knead dough on a floured surface for 15 minutes until smooth and elastic.",
            "Form into a ball, place in an oiled bowl, cover, and ferment for 2 hours.",
            "Divide dough into 4 equal portions. Shape into tight balls.",
            "Preheat oven with pizza stone to maximum temperature (at least 250°C, ideally 300°C).",
            "Stretch each dough ball by hand into a thin round disc (30cm diameter).",
            "Spread crushed tomatoes over the base, leaving a 1cm border.",
            "Distribute torn mozzarella pieces evenly.",
            "Transfer to pizza peel, slide onto hot stone, and bake for 6-7 minutes.",
            "Add fresh basil leaves and a drizzle of olive oil upon removal.",
            "Slice and serve immediately.",
        ],
        "safety": [
            "Pizza stone reaches extremely high temperatures - use pizza peel to transfer.",
            "Do not touch oven interior or stone without protection.",
            "Allow fermented dough to rest if it springs back excessively.",
            "Be cautious of steam when opening oven.",
        ],
    },
    "beef stir-fry": {
        "name": "Beef Stir-Fry",
        "cuisine": "Chinese",
        "difficulty": "beginner",
        "prep_time_minutes": 20,
        "cook_time_minutes": 10,
        "servings": 4,
        "ingredients": [
            {"item": "beef sirloin", "quantity": "500g", "prep": "thinly sliced against the grain"},
            {"item": "soy sauce", "quantity": "3 tbsp", "prep": "none"},
            {"item": "oyster sauce", "quantity": "2 tbsp", "prep": "none"},
            {"item": "cornstarch", "quantity": "1 tbsp", "prep": "dissolved in water"},
            {"item": "garlic", "quantity": "4 cloves", "prep": "minced"},
            {"item": "fresh ginger", "quantity": "2cm piece", "prep": "julienned"},
            {"item": "bell peppers", "quantity": "2", "prep": "sliced"},
            {"item": "broccoli", "quantity": "200g", "prep": "cut into florets"},
            {"item": "spring onions", "quantity": "4", "prep": "cut into 3cm lengths"},
            {"item": "vegetable oil", "quantity": "3 tbsp", "prep": "none"},
        ],
        "techniques": [
            {
                "name": "velveting",
                "description": "Marinate sliced beef with cornstarch and soy sauce to tenderise",
                "precision": "medium",
                "temperature_c": 4,
                "duration_minutes": 15,
            },
            {
                "name": "wok hei",
                "description": "Stir-fry over extremely high heat to achieve smoky flavour and seared edges",
                "precision": "high",
                "temperature_c": 300,
                "duration_minutes": 2,
            },
            {
                "name": "batch cooking",
                "description": "Cook ingredients in batches to maintain wok temperature",
                "precision": "high",
                "temperature_c": 300,
                "duration_minutes": 5,
            },
        ],
        "equipment": ["wok", "wok spatula", "chopping board", "sharp knife", "mixing bowls"],
        "temperatures": {
            "wok_heat": 300,
            "oil_shimmer": 200,
            "serving": 80,
        },
        "steps": [
            "Slice beef thinly against the grain. Marinate with soy sauce and cornstarch for 15 min.",
            "Prepare all vegetables before heating the wok (mise en place).",
            "Heat wok over the highest heat until it begins to smoke lightly.",
            "Add 2 tbsp oil, swirl to coat. Sear beef in a single layer for 60 seconds. Remove.",
            "Add remaining oil. Stir-fry garlic and ginger for 15 seconds.",
            "Add broccoli and peppers. Toss for 2 minutes.",
            "Return beef to wok. Add oyster sauce and toss everything together for 30 seconds.",
            "Add spring onions. Toss for a final 15 seconds.",
            "Serve immediately over steamed rice.",
        ],
        "safety": [
            "Wok reaches extremely high temperatures - keep a safe distance from the flame.",
            "Never add water to a hot oiled wok - it will splatter violently.",
            "Ensure all ingredients are dry before adding to hot oil.",
            "Use a long-handled spatula to avoid burns.",
        ],
    },
    "chocolate cake": {
        "name": "Chocolate Cake",
        "cuisine": "International",
        "difficulty": "intermediate",
        "prep_time_minutes": 25,
        "cook_time_minutes": 35,
        "servings": 12,
        "ingredients": [
            {"item": "plain flour", "quantity": "300g", "prep": "sifted"},
            {"item": "cocoa powder", "quantity": "75g", "prep": "sifted"},
            {"item": "caster sugar", "quantity": "350g", "prep": "none"},
            {"item": "eggs", "quantity": "3", "prep": "room temperature"},
            {"item": "buttermilk", "quantity": "240ml", "prep": "room temperature"},
            {"item": "vegetable oil", "quantity": "180ml", "prep": "none"},
            {"item": "hot water", "quantity": "240ml", "prep": "just boiled"},
            {"item": "baking soda", "quantity": "2 tsp", "prep": "none"},
            {"item": "baking powder", "quantity": "1 tsp", "prep": "none"},
            {"item": "vanilla extract", "quantity": "2 tsp", "prep": "none"},
            {"item": "dark chocolate", "quantity": "200g", "prep": "chopped (for ganache)"},
            {"item": "double cream", "quantity": "200ml", "prep": "heated (for ganache)"},
        ],
        "techniques": [
            {
                "name": "creaming and mixing",
                "description": "Combine wet and dry ingredients without overmixing",
                "precision": "medium",
                "temperature_c": 20,
                "duration_minutes": 5,
            },
            {
                "name": "baking",
                "description": "Bake until a skewer inserted in the centre comes out clean",
                "precision": "high",
                "temperature_c": 175,
                "duration_minutes": 35,
            },
            {
                "name": "ganache making",
                "description": "Pour hot cream over chopped chocolate and stir until smooth",
                "precision": "high",
                "temperature_c": 80,
                "duration_minutes": 5,
            },
            {
                "name": "frosting",
                "description": "Apply ganache evenly over cooled cake layers",
                "precision": "medium",
                "temperature_c": 30,
                "duration_minutes": 10,
            },
        ],
        "equipment": [
            "2 x 20cm cake tins",
            "mixing bowls",
            "electric mixer",
            "spatula",
            "wire cooling rack",
            "oven",
            "saucepan",
        ],
        "temperatures": {
            "oven": 175,
            "ganache_cream": 80,
            "ganache_setting": 20,
        },
        "steps": [
            "Preheat oven to 175°C. Grease and line two 20cm cake tins.",
            "Sift flour, cocoa powder, baking soda, and baking powder together.",
            "In a separate bowl, whisk sugar, eggs, buttermilk, oil, and vanilla.",
            "Fold dry ingredients into wet mixture until just combined.",
            "Stir in hot water (batter will be thin - this is correct).",
            "Divide batter between prepared tins.",
            "Bake for 30-35 minutes until a skewer comes out clean.",
            "Cool in tins for 10 minutes, then turn out onto wire racks.",
            "For ganache: heat cream to just below boiling, pour over chopped chocolate.",
            "Stir ganache until smooth and glossy. Cool until spreadable.",
            "Place one cake layer on a stand. Spread with ganache. Add second layer.",
            "Cover the entire cake with remaining ganache. Allow to set.",
        ],
        "safety": [
            "Hot water must be handled carefully when adding to batter.",
            "Allow cake to cool completely before applying ganache.",
            "Use oven gloves when handling hot tins.",
            "Check oven temperature with a thermometer for accurate baking.",
        ],
    },
    "fish and chips": {
        "name": "Fish and Chips",
        "cuisine": "British",
        "difficulty": "intermediate",
        "prep_time_minutes": 20,
        "cook_time_minutes": 30,
        "servings": 4,
        "ingredients": [
            {"item": "cod fillets", "quantity": "4 x 200g", "prep": "skinned and boned"},
            {"item": "plain flour", "quantity": "200g", "prep": "sifted"},
            {"item": "cornflour", "quantity": "50g", "prep": "none"},
            {"item": "cold sparkling water", "quantity": "250ml", "prep": "very cold"},
            {"item": "baking powder", "quantity": "1 tsp", "prep": "none"},
            {"item": "large potatoes", "quantity": "1kg", "prep": "cut into thick chips"},
            {"item": "vegetable oil", "quantity": "2 litres", "prep": "for deep frying"},
            {"item": "salt", "quantity": "to taste", "prep": "none"},
            {"item": "malt vinegar", "quantity": "to serve", "prep": "none"},
        ],
        "techniques": [
            {
                "name": "double frying chips",
                "description": "Fry chips twice: first to cook through, second to crisp",
                "precision": "high",
                "temperature_c": 160,
                "duration_minutes": 15,
            },
            {
                "name": "batter making",
                "description": "Mix cold sparkling water into flour for a light, crispy batter",
                "precision": "medium",
                "temperature_c": 4,
                "duration_minutes": 2,
            },
            {
                "name": "deep frying fish",
                "description": "Deep fry battered fish until golden and cooked through",
                "precision": "high",
                "temperature_c": 190,
                "duration_minutes": 6,
            },
        ],
        "equipment": [
            "deep fryer or large heavy pot",
            "cooking thermometer",
            "wire rack",
            "slotted spoon",
            "mixing bowl",
            "baking tray",
        ],
        "temperatures": {
            "first_fry_chips": 130,
            "second_fry_chips": 190,
            "fish_fry": 190,
            "oil_max_safe": 200,
        },
        "steps": [
            "Peel and cut potatoes into chips (1.5cm thick). Rinse in cold water and dry well.",
            "Heat oil to 130°C. Blanch chips for 6-8 minutes until cooked but not coloured. Drain.",
            "Mix flour, cornflour, baking powder, and a pinch of salt.",
            "Whisk in very cold sparkling water until batter is smooth but still slightly lumpy.",
            "Increase oil temperature to 190°C.",
            "Pat fish fillets dry. Dust with flour, then dip in batter.",
            "Carefully lower battered fish into hot oil. Fry for 5-6 minutes until golden.",
            "Remove fish to a wire rack. Keep warm in a low oven.",
            "Return chips to the 190°C oil. Fry for 3-4 minutes until golden and crisp.",
            "Drain chips on kitchen paper. Season with salt.",
            "Serve fish and chips immediately with malt vinegar.",
        ],
        "safety": [
            "Never fill fryer more than one-third with oil.",
            "Monitor oil temperature constantly - overheated oil can ignite.",
            "Ensure food is dry before lowering into hot oil to prevent splashing.",
            "Never leave hot oil unattended.",
            "Have a fire blanket or lid nearby in case of oil fire - never use water.",
        ],
    },
    "pad thai": {
        "name": "Pad Thai",
        "cuisine": "Thai",
        "difficulty": "intermediate",
        "prep_time_minutes": 25,
        "cook_time_minutes": 10,
        "servings": 4,
        "ingredients": [
            {"item": "flat rice noodles", "quantity": "250g", "prep": "soaked in warm water 20 min"},
            {"item": "prawns", "quantity": "200g", "prep": "peeled and deveined"},
            {"item": "firm tofu", "quantity": "150g", "prep": "pressed and cubed"},
            {"item": "eggs", "quantity": "2", "prep": "beaten"},
            {"item": "bean sprouts", "quantity": "150g", "prep": "washed"},
            {"item": "garlic chives", "quantity": "50g", "prep": "cut into 3cm lengths"},
            {"item": "garlic", "quantity": "3 cloves", "prep": "minced"},
            {"item": "tamarind paste", "quantity": "3 tbsp", "prep": "dissolved in water"},
            {"item": "fish sauce", "quantity": "3 tbsp", "prep": "none"},
            {"item": "palm sugar", "quantity": "2 tbsp", "prep": "grated"},
            {"item": "roasted peanuts", "quantity": "50g", "prep": "crushed"},
            {"item": "lime", "quantity": "2", "prep": "cut into wedges"},
        ],
        "techniques": [
            {
                "name": "noodle soaking",
                "description": "Soak dried rice noodles in warm water until pliable but not soft",
                "precision": "high",
                "temperature_c": 40,
                "duration_minutes": 20,
            },
            {
                "name": "sauce balancing",
                "description": "Balance tamarind, fish sauce, and palm sugar for sweet-sour-salty harmony",
                "precision": "critical",
                "temperature_c": 20,
                "duration_minutes": 3,
            },
            {
                "name": "high-heat wok frying",
                "description": "Cook at high heat in a wok, tossing ingredients rapidly",
                "precision": "high",
                "temperature_c": 250,
                "duration_minutes": 5,
            },
        ],
        "equipment": ["wok", "wok spatula", "mixing bowls", "chopping board", "sharp knife"],
        "temperatures": {
            "noodle_soak": 40,
            "wok_heat": 250,
            "serving": 80,
        },
        "steps": [
            "Soak rice noodles in warm water for 20 minutes. Drain.",
            "Mix tamarind paste, fish sauce, and palm sugar to make pad thai sauce.",
            "Heat wok over high heat. Add oil, fry tofu cubes until golden. Remove.",
            "Add more oil. Stir-fry garlic for 10 seconds, then add prawns. Cook 1 minute.",
            "Push prawns to one side. Pour beaten eggs into wok. Scramble lightly.",
            "Add drained noodles and pad thai sauce. Toss for 1-2 minutes.",
            "Return tofu. Add half the bean sprouts and garlic chives. Toss 30 seconds.",
            "Plate and top with remaining bean sprouts, crushed peanuts, and lime wedges.",
        ],
        "safety": [
            "Ensure prawns are thoroughly cooked (pink and opaque throughout).",
            "Wok reaches very high temperatures - use long-handled utensils.",
            "Drain noodles well to prevent oil splatter.",
            "Keep workspace clear due to rapid cooking pace.",
        ],
    },
    "french omelette": {
        "name": "French Omelette",
        "cuisine": "French",
        "difficulty": "advanced",
        "prep_time_minutes": 5,
        "cook_time_minutes": 3,
        "servings": 1,
        "ingredients": [
            {"item": "eggs", "quantity": "3", "prep": "beaten thoroughly"},
            {"item": "butter", "quantity": "15g", "prep": "none"},
            {"item": "salt", "quantity": "pinch", "prep": "none"},
            {"item": "white pepper", "quantity": "pinch", "prep": "none"},
            {"item": "fresh herbs", "quantity": "1 tbsp", "prep": "finely chopped (chives, tarragon)"},
        ],
        "techniques": [
            {
                "name": "egg beating",
                "description": "Beat eggs vigorously until yolks and whites are fully homogeneous",
                "precision": "medium",
                "temperature_c": 20,
                "duration_minutes": 1,
            },
            {
                "name": "pan shaking",
                "description": "Continuously shake pan while stirring to create small, creamy curds",
                "precision": "critical",
                "temperature_c": 150,
                "duration_minutes": 1,
            },
            {
                "name": "rolling",
                "description": "Roll omelette onto plate in a smooth, seamless motion",
                "precision": "critical",
                "temperature_c": 65,
                "duration_minutes": 0.5,
            },
        ],
        "equipment": [
            "non-stick pan (20cm)",
            "fork or chopstick",
            "plate",
        ],
        "temperatures": {
            "butter_foam": 150,
            "cooking": 150,
            "centre_target": 65,
        },
        "steps": [
            "Crack 3 eggs into a bowl. Season with salt and white pepper.",
            "Beat vigorously with a fork until completely homogeneous.",
            "Heat a 20cm non-stick pan over medium-high heat.",
            "Add butter. When it foams and the foam begins to subside, add eggs.",
            "Immediately stir with a fork or chopstick while shaking the pan.",
            "When the eggs are 80% set with a creamy, barely-set centre, stop stirring.",
            "Tilt pan and fold one-third of the omelette over itself.",
            "Slide onto a plate, rolling the omelette to seal the fold underneath.",
            "Rub surface with a small knob of butter for a glossy finish.",
            "Garnish with fresh herbs. Serve immediately.",
        ],
        "safety": [
            "Pan and butter are very hot - avoid touching the cooking surface.",
            "Work quickly once eggs hit the pan - the entire cook takes 60-90 seconds.",
            "Do not overcook; the centre should remain slightly baveuse (creamy).",
        ],
    },
    "bread": {
        "name": "Artisan Bread",
        "cuisine": "International",
        "difficulty": "intermediate",
        "prep_time_minutes": 30,
        "cook_time_minutes": 45,
        "servings": 1,
        "ingredients": [
            {"item": "strong bread flour", "quantity": "500g", "prep": "none"},
            {"item": "water", "quantity": "350ml", "prep": "lukewarm (37°C)"},
            {"item": "salt", "quantity": "10g", "prep": "fine"},
            {"item": "instant yeast", "quantity": "7g", "prep": "none"},
            {"item": "olive oil", "quantity": "1 tbsp", "prep": "optional"},
        ],
        "techniques": [
            {
                "name": "kneading",
                "description": "Knead dough to develop gluten until smooth and passing the windowpane test",
                "precision": "high",
                "temperature_c": 25,
                "duration_minutes": 12,
            },
            {
                "name": "bulk fermentation",
                "description": "First rise to double in size at room temperature",
                "precision": "medium",
                "temperature_c": 25,
                "duration_minutes": 90,
            },
            {
                "name": "shaping",
                "description": "Shape dough into a boule or batard with good surface tension",
                "precision": "high",
                "temperature_c": 25,
                "duration_minutes": 5,
            },
            {
                "name": "proofing",
                "description": "Final proof until the dough passes the poke test",
                "precision": "high",
                "temperature_c": 25,
                "duration_minutes": 45,
            },
            {
                "name": "steam baking",
                "description": "Bake with steam for oven spring and crust development",
                "precision": "critical",
                "temperature_c": 230,
                "duration_minutes": 40,
            },
        ],
        "equipment": [
            "mixing bowl",
            "bench scraper",
            "banneton or bowl with cloth",
            "dutch oven or baking stone",
            "oven",
            "lame or sharp knife (for scoring)",
            "spray bottle",
        ],
        "temperatures": {
            "water": 37,
            "fermentation": 25,
            "oven_initial": 230,
            "oven_reduced": 210,
        },
        "steps": [
            "Combine flour, salt, and yeast in a large bowl.",
            "Add lukewarm water (37°C) and mix until a shaggy dough forms.",
            "Turn out onto an unfloured surface. Knead for 10-12 minutes until smooth and elastic.",
            "Perform the windowpane test: stretch a small piece thin enough to see light through.",
            "Place in a lightly oiled bowl, cover. Ferment for 90 minutes or until doubled.",
            "Turn out dough. Degas gently. Pre-shape into a round. Rest 15 minutes.",
            "Shape into a boule with good surface tension. Place seam-side up in a floured banneton.",
            "Proof for 45 minutes. Preheat oven to 230°C with a Dutch oven inside.",
            "Turn dough into the hot Dutch oven. Score the top with a lame or sharp knife.",
            "Cover and bake for 25 minutes (steam phase).",
            "Remove lid. Reduce heat to 210°C. Bake for 15-20 minutes until deep golden brown.",
            "The bread should sound hollow when tapped on the bottom. Cool on a wire rack for 1 hour.",
        ],
        "safety": [
            "Dutch oven is extremely hot (230°C) - always use oven gloves.",
            "Scoring requires a very sharp blade - cut away from your body.",
            "Allow bread to cool fully before slicing to complete the internal cooking.",
            "Be cautious of steam when removing the Dutch oven lid.",
        ],
    },
}

# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def analyse_dish(dish_name: str) -> str:
    """
    Analyse a dish and return structured information including ingredients,
    techniques, equipment, temperatures, and step-by-step instructions.

    Args:
        dish_name: Name of the dish to analyse (e.g. 'pasta carbonara', 'souffle')
    """
    key = dish_name.lower().strip()

    # Try exact match first, then partial match
    dish = DISHES.get(key)
    if dish is None:
        for db_key, db_dish in DISHES.items():
            if key in db_key or db_key in key:
                dish = db_dish
                break

    if dish is None:
        available = ", ".join(sorted(DISHES.keys()))
        return json.dumps(
            {
                "error": f"Dish '{dish_name}' not found in database.",
                "available_dishes": available,
                "suggestion": "Try one of the available dishes listed above.",
            },
            indent=2,
        )

    return json.dumps(dish, indent=2)


@mcp.tool()
def get_cooking_techniques(dish_name: str) -> str:
    """
    Get a detailed breakdown of the cooking techniques required for a dish,
    including precision requirements, temperatures, and durations.

    Args:
        dish_name: Name of the dish (e.g. 'sushi rolls', 'beef stir-fry')
    """
    key = dish_name.lower().strip()

    dish = DISHES.get(key)
    if dish is None:
        for db_key, db_dish in DISHES.items():
            if key in db_key or db_key in key:
                dish = db_dish
                break

    if dish is None:
        available = ", ".join(sorted(DISHES.keys()))
        return json.dumps(
            {
                "error": f"Dish '{dish_name}' not found.",
                "available_dishes": available,
            },
            indent=2,
        )

    result = {
        "dish": dish["name"],
        "difficulty": dish["difficulty"],
        "techniques": dish["techniques"],
        "total_techniques": len(dish["techniques"]),
        "critical_techniques": [
            t for t in dish["techniques"] if t["precision"] == "critical"
        ],
        "temperature_range": {
            "min_c": min(t["temperature_c"] for t in dish["techniques"]),
            "max_c": max(t["temperature_c"] for t in dish["techniques"]),
        },
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def get_equipment_specs(equipment_name: str) -> str:
    """
    Get specifications for kitchen equipment including typical operating
    temperatures, power requirements, and usage notes.

    Args:
        equipment_name: Name of the equipment (e.g. 'oven', 'wok', 'deep fryer')
    """
    EQUIPMENT_DB = {
        "oven": {
            "name": "Commercial Convection Oven",
            "type": "heating",
            "temperature_range_c": {"min": 50, "max": 300},
            "power_watts": 5000,
            "dimensions_cm": {"width": 75, "height": 70, "depth": 65},
            "features": [
                "convection fan",
                "top and bottom heating elements",
                "temperature probe",
                "timer",
                "steam injection",
            ],
            "precision_c": 5,
            "notes": "Preheat for at least 20 minutes for accurate temperature.",
        },
        "wok": {
            "name": "Carbon Steel Wok",
            "type": "cookware",
            "temperature_range_c": {"min": 100, "max": 350},
            "diameter_cm": 36,
            "material": "carbon steel",
            "features": [
                "round bottom",
                "single long handle",
                "rapid heat transfer",
                "seasoned surface",
            ],
            "heat_source": "high-BTU gas burner recommended",
            "notes": "Requires seasoning. Reaches very high temperatures for wok hei.",
        },
        "deep fryer": {
            "name": "Commercial Deep Fryer",
            "type": "heating",
            "temperature_range_c": {"min": 120, "max": 200},
            "capacity_litres": 10,
            "power_watts": 3500,
            "features": [
                "thermostat control",
                "safety cutoff",
                "basket lift",
                "oil drain valve",
                "temperature display",
            ],
            "precision_c": 2,
            "notes": "Never exceed 200°C. Monitor oil quality regularly.",
        },
        "electric mixer": {
            "name": "Stand Mixer",
            "type": "mechanical",
            "speed_range_rpm": {"min": 50, "max": 300},
            "power_watts": 800,
            "capacity_litres": 5,
            "attachments": ["whisk", "paddle", "dough hook"],
            "features": ["variable speed", "tilt head", "splash guard"],
            "notes": "Start on low speed to prevent splashing. Ideal for whipping egg whites.",
        },
        "rice cooker": {
            "name": "Programmable Rice Cooker",
            "type": "heating",
            "temperature_range_c": {"min": 60, "max": 105},
            "capacity_litres": 3,
            "power_watts": 700,
            "features": [
                "fuzzy logic control",
                "keep warm function",
                "timer delay",
                "multiple rice settings",
            ],
            "precision_c": 1,
            "notes": "Precise water-to-rice ratio is essential.",
        },
        "non-stick pan": {
            "name": "Non-Stick Frying Pan (20cm)",
            "type": "cookware",
            "temperature_range_c": {"min": 100, "max": 200},
            "diameter_cm": 20,
            "material": "aluminium with PTFE coating",
            "features": [
                "non-stick surface",
                "induction-compatible base",
                "ergonomic handle",
            ],
            "notes": "Do not exceed 200°C or use metal utensils. Ideal for omelettes and delicate work.",
        },
        "dutch oven": {
            "name": "Cast Iron Dutch Oven",
            "type": "cookware",
            "temperature_range_c": {"min": 100, "max": 260},
            "capacity_litres": 5.5,
            "material": "enamelled cast iron",
            "features": [
                "excellent heat retention",
                "oven-safe",
                "tight-fitting lid for steam",
            ],
            "notes": "Preheat in oven for bread baking. Very heavy - handle with care.",
        },
        "saucepan": {
            "name": "Stainless Steel Saucepan",
            "type": "cookware",
            "temperature_range_c": {"min": 60, "max": 250},
            "capacity_litres": 3,
            "material": "tri-ply stainless steel",
            "features": ["graduated interior", "pouring lip", "helper handle"],
            "notes": "Good for béchamel, ganache, and general sauce work.",
        },
    }

    key = equipment_name.lower().strip()
    equipment = EQUIPMENT_DB.get(key)
    if equipment is None:
        for db_key, db_equip in EQUIPMENT_DB.items():
            if key in db_key or db_key in key:
                equipment = db_equip
                break

    if equipment is None:
        available = ", ".join(sorted(EQUIPMENT_DB.keys()))
        return json.dumps(
            {
                "error": f"Equipment '{equipment_name}' not found.",
                "available_equipment": available,
            },
            indent=2,
        )

    return json.dumps(equipment, indent=2)


@mcp.tool()
def get_safety_requirements(dish_name: str) -> str:
    """
    Get safety considerations and requirements for preparing a specific dish,
    including temperature hazards, allergens, and handling precautions.

    Args:
        dish_name: Name of the dish (e.g. 'fish and chips', 'chocolate cake')
    """
    key = dish_name.lower().strip()

    dish = DISHES.get(key)
    if dish is None:
        for db_key, db_dish in DISHES.items():
            if key in db_key or db_key in key:
                dish = db_dish
                break

    if dish is None:
        available = ", ".join(sorted(DISHES.keys()))
        return json.dumps(
            {
                "error": f"Dish '{dish_name}' not found.",
                "available_dishes": available,
            },
            indent=2,
        )

    # Compile comprehensive safety information
    max_temp = max(dish["temperatures"].values())
    result = {
        "dish": dish["name"],
        "safety_warnings": dish["safety"],
        "temperature_hazards": {
            "max_temperature_c": max_temp,
            "high_temp_steps": [
                t["name"]
                for t in dish["techniques"]
                if t["temperature_c"] >= 150
            ],
        },
        "equipment_requiring_care": dish["equipment"],
        "critical_techniques": [
            {
                "technique": t["name"],
                "description": t["description"],
                "temperature_c": t["temperature_c"],
            }
            for t in dish["techniques"]
            if t["precision"] == "critical"
        ],
        "general_kitchen_safety": [
            "Always have a first aid kit accessible.",
            "Ensure fire extinguisher is within reach.",
            "Wear appropriate protective equipment (oven gloves, apron).",
            "Keep work surfaces clean and dry.",
            "Handle sharp knives with care - always cut away from body.",
        ],
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Run as MCP server
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Budget & Nutrition Tools (for Smart Budget RobotChef)
# ---------------------------------------------------------------------------

# Simulated nutrition and price data (per dish, per serving)
DISH_NUTRITION = {
    "pasta carbonara": {"protein": 29, "kcal": 620, "vegetarian": False, "price_per_serving": 2.60},
    "cheese soufflé": {"protein": 18, "kcal": 410, "vegetarian": True, "price_per_serving": 2.80},
    "sushi rolls": {"protein": 16, "kcal": 350, "vegetarian": False, "price_per_serving": 3.10},
    "pizza margherita": {"protein": 14, "kcal": 480, "vegetarian": True, "price_per_serving": 2.20},
    "beef stir-fry": {"protein": 32, "kcal": 540, "vegetarian": False, "price_per_serving": 3.00},
    "chocolate cake": {"protein": 5, "kcal": 390, "vegetarian": True, "price_per_serving": 1.10},
    "fish and chips": {"protein": 28, "kcal": 700, "vegetarian": False, "price_per_serving": 3.50},
    "pad thai": {"protein": 15, "kcal": 520, "vegetarian": False, "price_per_serving": 2.70},
    "french omelette": {"protein": 17, "kcal": 250, "vegetarian": True, "price_per_serving": 1.50},
    "bread": {"protein": 9, "kcal": 210, "vegetarian": True, "price_per_serving": 0.60},
}

@mcp.tool()
def search_dishes(budget: float, nutrition: str, people: int) -> str:
    """
    Search for a dish that fits the given budget, nutrition target, and number of people.
    And then return the best match with justification, price and nutritional info.

    Args:
        budget: Total budget in GBP (e.g. 12.0)
        nutrition: Nutrition target ('high protein', 'vegetarian', 'balanced')
        people: Number of servings required
    """
    nutrition = nutrition.lower().strip()
    candidates = []
    for key, info in DISH_NUTRITION.items():
        total_price = info["price_per_serving"] * people
        if total_price > budget:
            continue
        if nutrition == "vegetarian" and not info["vegetarian"]:
            continue
        if nutrition == "high protein" and info["protein"] < 20:
            continue
        candidates.append((key, info, total_price))

    # If no strict match, relax protein threshold for high protein
    if not candidates and nutrition == "high protein":
        for key, info in DISH_NUTRITION.items():
            total_price = info["price_per_serving"] * people
            if total_price > budget:
                continue
            if info["protein"] >= 15:
                candidates.append((key, info, total_price))

    # If still no match, allow any dish under budget
    if not candidates:
        for key, info in DISH_NUTRITION.items():
            total_price = info["price_per_serving"] * people
            if total_price <= budget:
                candidates.append((key, info, total_price))

    if not candidates:
        return json.dumps({
            "error": f"No dish found for £{budget} ({people} people, {nutrition})",
            "suggestion": "Try increasing budget or relaxing nutrition target."
        }, indent=2)

    # Pick the highest protein (if high protein), else lowest price, else first
    if nutrition == "high protein":
        best = max(candidates, key=lambda x: x[1]["protein"])
    elif nutrition == "vegetarian":
        best = min(candidates, key=lambda x: x[2])
    else:
        best = min(candidates, key=lambda x: abs(x[1]["protein"] - 15) + x[2])

    key, info, total_price = best
    dish = DISHES[key]
    return json.dumps({
        "dish_name": dish["name"],
        "price": round(total_price, 2),
        "protein_per_serving": info["protein"],
        "kcal_per_serving": info["kcal"],
        "vegetarian": info["vegetarian"],
        "servings": people,
        "justification": f"Selected for {nutrition} within £{budget} budget. Protein: {info['protein']}g/serving, Price: £{round(total_price,2)} for {people} people."
    }, indent=2)


@mcp.tool()
def get_nutrition(dish_name: str, people: int) -> str:
    """
    Get nutrition info like protein, kcal for a dish and number of people.
    Robust to name variations (partial/case-insensitive match).
    """
    key = dish_name.lower().strip()
    info = DISH_NUTRITION.get(key)
    if not info:
        # Try partial/case-insensitive match
        for db_key, db_info in DISH_NUTRITION.items():
            if key in db_key or db_key in key:
                info = db_info
                break
    if not info:
        return json.dumps({"error": f"No nutrition info for {dish_name}"}, indent=2)
    return json.dumps({
        "dish_name": dish_name,
        "protein_per_serving": info["protein"],
        "kcal_per_serving": info["kcal"],
        "total_protein": info["protein"] * people,
        "total_kcal": info["kcal"] * people,
        "servings": people,
    }, indent=2)


@mcp.tool()
def get_price(dish_name: str, people: int) -> str:
    """
    Get realistic price for a dish for the given number of people.
    Robust to name variations (partial/case-insensitive match).
    """
    key = dish_name.lower().strip()
    info = DISH_NUTRITION.get(key)
    if not info:
        # Try partial/case-insensitive match
        for db_key, db_info in DISH_NUTRITION.items():
            if key in db_key or db_key in key:
                info = db_info
                break
    if not info:
        return json.dumps({"error": f"No price info for {dish_name}"}, indent=2)
    return json.dumps({
        "dish_name": dish_name,
        "price_per_serving": info["price_per_serving"],
        "total_price": round(info["price_per_serving"] * people, 2),
        "servings": people,
    }, indent=2)

if __name__ == "__main__":
    mcp.run()
