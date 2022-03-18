# Konwolucyjne sieci neuronowe
1. Sieci konwolucyjne działają na trójwymiarowych tensorach określanych mianem map cech (dwie przestrzenne osie definiujące wysokość i szerokość oraz jedna oś - oś głębi/oś kanałów).
2. Warstwy konwolucyjne uczą się lokalnych wzorców
3. Konwolucja polega na przesuwaniu okien o wymiarach 3x3 lub 5x5 po trójwymiarowej mapie cech.
4. Krok - parametr konwolucji, czynnik wpływający na rozmiar mapy wyjściowej. Jest to odległość między dwoma kolejnymi oknami. Domyślna jego wartość to 1, jednak można przypisać mu większą wartość, co doprowadzi do uzyskania tzw. konwolucji kroczących. Przypisanie parametrowi kroku wartości 2 oznacza, że szerokość i wysokość mapy cech są poddawane skalowaniu (są zmniejszane dwukrotnie).
5. Operacja skalowania max-pooling polega na przeskalowaniu mapy cech zamiast kroków. Polega na agresywnym zmniejszaniu rozdzielczości map cech, działa bardzo podobnie do konwolucji kroczących.
6. Technika augmentacji danych pozwala na zmniejszenie skutków nadmiernego dopasowania modelu.