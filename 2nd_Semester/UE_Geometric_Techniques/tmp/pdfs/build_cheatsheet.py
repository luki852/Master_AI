
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

data = """@1-6|Learning in high dimensions|Lernen in hohen Dimensionen
Supervised learning|Überwachtes Lernen
i.i.d. pairs (X,Y). True risk R(f)=E[ℓ(f(X),Y)]; Bayes predictor minimises it over all measurable f (squared loss: E[Y∣X]). ERM minimises sample-average loss over a model class H. Complexity: NN width/depth, norms or implicit bias; Sobolev smoothness and norm bound.|i.i.d. Paare (X,Y). Wahres Risiko R(f)=E[ℓ(f(X),Y)]; Bayes-Prädiktor minimiert es über alle messbaren f (quadratischer Verlust: E[Y∣X]). ERM minimiert den mittleren Trainingsverlust in H. Komplexität: Breite/Tiefe, Normen bzw. impliziter Bias; bei Sobolev-Klassen Glattheit und Normschranke.
Error and double descent|Fehler und Double Descent
Excess risk = approximation gap (best in H vs Bayes) + estimation gap (exact ERM vs best in H) + optimisation effect (trained fit vs exact ERM in true risk; commonly bounded via empirical suboptimality). With complexity, approximation falls and classical estimation tends to rise. Test-error sketch: U before interpolation, peak near first zero train error, then second descent. Implicit regularisation can favour low-norm interpolants. More data typically shifts interpolation right.|Exzessrisiko = Approximationslücke (bestes Modell in H vs. Bayes) + Schätzlücke (exaktes ERM vs. bestes Modell in H) + Optimierungseffekt (trainiertes Modell vs. exaktes ERM im wahren Risiko; oft über empirische Suboptimalität abgeschätzt). Mit Komplexität sinkt Approximation, klassisch steigt die Schätzkomponente tendenziell. Testfehler: U-Form, Spitze bei erstmals null Trainingsfehler, danach zweiter Abstieg. Implizite Regularisierung kann Interpolanten kleiner Norm bevorzugen. Mehr Daten verschieben die Interpolationsschwelle meist nach rechts.
Curse of dimensionality|Fluch der Dimensionalität
Local mass scales as rᵈ, so nearby samples are scarce. Put disjoint Lipschitz bumps with independent signs on the 2ᵈ cube corners. Unseen corners have unpredictable signs → exponentially many samples. Slide comparison: Lipschitz bottleneck is estimation; Sobolev approximation error can decay as m^(−s/d) for m parameters and smoothness s. Manifold hypothesis: data near a low-dimensional manifold (intrinsic dimension governs local covering).|Lokale Masse skaliert wie rᵈ, nahe Datenpunkte sind daher selten. Platziere disjunkte Lipschitz-Buckel mit unabhängigen Vorzeichen an 2ᵈ Würfelecken. Vorzeichen ungesehener Ecken sind unvorhersagbar → exponentieller Datenbedarf. Folienvergleich: Lipschitz-Klasse → Schätzfehler; Sobolev-Klasse → Approximationsfehler kann wie m^(−s/d) abfallen (m Parameter, Glattheit s). Mannigfaltigkeitshypothese: Daten nahe einer niedrigdimensionalen Mannigfaltigkeit, deren Dimension die lokale Überdeckung bestimmt.
Loss landscape|Verlustlandschaft
Generic nonconvex: valleys, bad minima and many saddles. Deep nets often exhibit broad, connected low-loss regions and many equivalent solutions (e.g. neuron permutations); no general guarantee of global optimality.|Allgemein nichtkonvex: Täler, schlechte Minima, viele Sattelpunkte. Tiefe Netze zeigen oft breite, verbundene Bereiche mit kleinem Loss und äquivalente Lösungen (z. B. durch Neuronenpermutation). Globale Optimalität ist im Allgemeinen nicht garantiert.
@7-15|Geometric priors|Geometrische Priors
Signals, groups and actions|Signale, Gruppen und Wirkungen
Signal x: Ω→ℝᶜ; pointwise operations give a vector space. Group: closed, associative product, identity e and inverses. Homomorphism φ(gh)=φ(g)φ(h); bijective ⇒ isomorphism. det: GL(n,ℝ)→ℝ∖{0}; exp: (ℝ,+)≅(ℝ₊,×) via exp(a+b)=exp(a)exp(b).|Signal x: Ω→ℝᶜ; punktweise Operationen ergeben einen Vektorraum. Gruppe: abgeschlossenes, assoziatives Produkt, neutrales Element e und Inverse. Homomorphismus φ(gh)=φ(g)φ(h); bijektiv ⇒ Isomorphismus. det: GL(n,ℝ)→ℝ∖{0}; exp: (ℝ,+)≅(ℝ₊,×) wegen exp(a+b)=exp(a)exp(b).
Action: e·u=u, g·(h·u)=(gh)·u. Pullback ρ(g)x(u)=x(g⁻¹·u) gives linear, invertible maps with ρ(gh)=ρ(g)ρ(h). The inverse reverses pullback order.|Wirkung: e·u=u, g·(h·u)=(gh)·u. Pullback ρ(g)x(u)=x(g⁻¹·u) liefert lineare, invertierbare Abbildungen mit ρ(gh)=ρ(g)ρ(h). Das Inverse kehrt die Reihenfolge beim Pullback um.
Equivariance and invariance|Äquivarianz und Invarianz
Equivariance fρₓ(g)=ρᵧ(g)f: x→f(x)→ρᵧ(g)f(x) agrees with x→ρₓ(g)x→f(ρₓ(g)x). Each input orbit maps into one output orbit; distinct orbits may merge. Invariance: trivial output action, constant on each orbit.|Äquivarianz fρₓ(g)=ρᵧ(g)f: x→f(x)→ρᵧ(g)f(x) stimmt mit x→ρₓ(g)x→f(ρₓ(g)x) überein. Jede Eingabebahn geht in eine Ausgabebahn über; verschiedene Orbits können zusammenfallen. Invarianz: triviale Ausgabewirkung, konstant pro Orbit.
Finite G: (Sᴳf)(x)=∣G∣⁻¹ Σ[g∈G] f(ρ(g)x). Orthogonal projection under invariant input law in L². For invariant target, approximation cannot worsen (equal for averaging-closed H); estimation can improve through lower complexity. High-dimensional orbit space can still be cursed.|Endliches G: (Sᴳf)(x)=∣G∣⁻¹ Σ[g∈G] f(ρ(g)x). Bei invarianter Eingabeverteilung orthogonale Projektion in L². Für invariantes Ziel wird Approximation nicht schlechter (gleich bei mittelungsabgeschlossenem H). Schätzung kann durch geringere Komplexität profitieren. Ein hochdimensionaler Orbitraum bleibt problematisch.
Blueprint and stability|Blueprint und Stabilität
Repeat local equivariant layers + nonlinearities → local pooling/coarsening → repeat at coarser scale → optional global invariant pooling → task head. Seek expressive, sample-efficient, stable, scalable classes. Local pooling removes fine detail and builds context; global pooling collapses the domain. Shared weights and multiscale processing reinforce each other. Sketch Tx close to gx on an orbit: invariant f gives f(gx)=f(x); stability keeps ‖f(Tx)−f(x)‖ small as the deviation grows from zero.|Lokale äquivariante Schichten + Nichtlinearitäten → lokales Pooling/Vergröberung → Wiederholung auf gröberer Skala → optional globales invariantes Pooling → Task-Head. Ziel: ausdrucksstark, dateneffizient, stabil und skalierbar. Lokales Pooling reduziert Details und erweitert den Kontext, globales Pooling reduziert den ganzen Definitionsbereich. Gewichtsbindung und mehrere Skalen ergänzen sich. Skizze: Tx nahe gx auf einer Bahn. Invariantes f erfüllt f(gx)=f(x), Stabilität hält ‖f(Tx)−f(x)‖ bei kleiner Abweichung klein.
||
@16-19|Sets|Mengen
Permutation action and linear maps|Permutationswirkung und lineare Abbildungen
X∈ℝⁿˣᶜ with elements in rows. G=Sₙ, Pᵢⱼ=1 iff i=π(j), so (PX)ᵢ=X[π⁻¹(i)]. Linear equivariance L(PX)=PL(X). For one channel, commuting matrices have constant diagonal and constant off-diagonal: aI+b11ᵀ. Multichannel: Yᵢ=XᵢW+ΣⱼXⱼV.|X∈ℝⁿˣᶜ mit Elementen als Zeilen. G=Sₙ, Pᵢⱼ=1 genau wenn i=π(j), also (PX)ᵢ=X[π⁻¹(i)]. Linear äquivariant: L(PX)=PL(X). Ein Kanal: kommutierende Matrizen haben konstante Diagonale und konstante Nebendiagonaleinträge, aI+b11ᵀ. Mehrere Kanäle: Yᵢ=XᵢW+ΣⱼXⱼV.
DeepSets: ρ(Σᵢφ(Xᵢ)). Shared learnable φ is equivariant; fixed sum is invariant; learnable ρ maps pooled features to the output.|DeepSets: ρ(Σᵢφ(Xᵢ)). Geteiltes lernbares φ ist äquivariant, feste Summe invariant. Lernbares ρ bildet die gepoolten Merkmale auf die Ausgabe ab.
@20-30|Graphs|Graphen
Graph and relabelling|Graph und Umnummerierung
Undirected node-attributed graph (V,E,X), E unordered pairs; symmetric adjacency A, node matrix X and optional edge tensor B. Edges define neighbourhoods. Act by Sₙ: X→PX, A→PAPᵀ, Bₖ→PBₖPᵀ per channel k. Node output transforms by P; graph output is invariant.|Ungerichteter Graph (V,E,X) mit Knotenmerkmalen und ungeordneten Kantenpaaren. Symmetrische Adjazenz A, Knotenmatrix X, optional Kantentensor B. Kanten definieren Nachbarschaften. Sₙ-Wirkung: X→PX, A→PAPᵀ, Bₖ→PBₖPᵀ je Kanal k. Knotenausgabe permutiert mit P, Graphausgabe ist invariant.
Message passing|Message Passing
mᵢ=AGG{M(hᵢ,hⱼ,eᵢⱼ): j∈N(i)}, hᵢ′=U(hᵢ,mᵢ). Shared M,U and symmetric AGG ensure equivariance. Graph conv: fixed graph weights times shared feature transform, e.g. σ(ÂXW) with normalised adjacency incl. self-loops. Attention learns scalar weights; general MPNN learns vector messages. Global sum/mean gives invariant readout.|mᵢ=AGG{M(hᵢ,hⱼ,eᵢⱼ): j∈N(i)}, hᵢ′=U(hᵢ,mᵢ). Geteilte M,U und symmetrische AGG sichern Äquivarianz. Graph Conv: feste Graphgewichte und geteilte Merkmalstransformation, z. B. σ(ÂXW) mit normierter Adjazenz inkl. Selbstschleifen. Attention lernt skalare Gewichte, allgemeines MPNN Vektornachrichten. Globale Summe/Mittel liefert invarianten Readout.
Example: path 1-2-3, h=(2,4,7). Sum of neighbours gives (4,9,4); adding self gives (6,13,11). Swap 1 and 3 → (11,13,6) in new row order.|Beispiel: Pfad 1-2-3, h=(2,4,7). Nachbarsumme ergibt (4,9,4); mit Eigenwert des Knotens (6,13,11). Tausch 1 und 3 → (11,13,6) in neuer Zeilenreihenfolge.
1-WL colour refinement|1-WL-Farbverfeinerung
Update all nodes simultaneously: cᵢ′=HASH(cᵢ, multiset{cⱼ:j∈N(i)}), using the same injective dictionary across graphs. From uniform labels, round 1 colours encode degree. P₄ has counts (2,2) for degrees 1,2; K₁,₃ has counts (3,1) for 1,3 → non-isomorphic. Equal histograms are inconclusive (C₆ vs two triangles). Standard MPNNs ≤ 1-WL; injective multiset aggregation and update, plus injective readout for graph tasks, can match it. Mean loses multiplicity.|Simultan aktualisieren: cᵢ′=HASH(cᵢ, Multimenge{cⱼ:j∈N(i)}), mit demselben injektiven Wörterbuch für beide Graphen. Bei gleichen Startfarben kodiert Runde 1 den Grad. P₄ hat Häufigkeiten (2,2) für Grad 1,2; K₁,₃ hat (3,1) für 1,3 → nicht isomorph. Gleiche Histogramme sind nicht entscheidend (C₆ vs. zwei Dreiecke). Standard-MPNNs ≤ 1-WL. Injektive Multimengenaggregation und Updates, für Graphausgaben auch injektiver Readout, erreichen 1-WL. Mittelwert verliert Vielfachheiten.
Beyond local MPNNs|Erweiterungen lokaler MPNNs
Positional encodings add structural coordinates/distances; handle spectral sign/basis ambiguity. Substructure networks count motifs (e.g. triangles). Subgraph GNNs encode an equivariantly chosen collection of subgraphs and pool. Reconstruction conjecture: an undirected simple graph with ≥3 vertices is determined up to isomorphism by its multiset of vertex-deleted graphs.|Positionskodierungen ergänzen Strukturkoordinaten/Distanzen; spektrale Vorzeichen- und Basismehrdeutigkeit beachten. Substructure Networks zählen Motive (z. B. Dreiecke). Subgraph-GNNs kodieren äquivariant gewählte Teilgraphen und poolen. Rekonstruktionsvermutung: Ein einfacher ungerichteter Graph mit ≥3 Knoten ist bis auf Isomorphie durch die Multimenge der Graphen nach Löschen je eines Knotens bestimmt.
Transformer: self-attention is message passing on a complete graph (softmax-weighted values). Bare self-attention is permutation-equivariant; fixed positions or a causal mask break arbitrary permutation symmetry unless carried along.|Transformer: Self-Attention ist Message Passing auf einem vollständigen Graphen (Softmax-gewichtete Values). Reine Self-Attention ist permutationsäquivariant. Feste Positionen oder kausale Maske brechen beliebige Permutationssymmetrie, sofern sie nicht mitpermutiert werden.
Oversquashing|Oversquashing
Many distant signals are compressed into fixed-width states through bottlenecks: ‖∂hᵢᴸ/∂hⱼ⁰‖ small, bounded in typical MPNNs by a layer factor times [(I+A)ᴸ]ᵢⱼ (or normalised analogue). Width can help; extra depth alone is insufficient. Topology dominates (high effective resistance, small cuts); rewiring can open routes.|Viele entfernte Signale werden durch Engpässe in Zustände fester Breite komprimiert: ‖∂hᵢᴸ/∂hⱼ⁰‖ klein, in typischen MPNNs beschränkt durch einen Schichtfaktor mal [(I+A)ᴸ]ᵢⱼ (oder normiertes Analogon). Mehr Breite kann helfen. Tiefe allein reicht kaum. Topologie dominiert (hoher effektiver Widerstand, kleine Schnitte); Rewiring kann Wege öffnen.
||
@31-37|Grids and wavelets|Gitter und Wavelets
Shifts and linear convolution|Verschiebung und lineare Faltung
Periodic 1D grid = cycle; translations ℤₙ act by (Tₛx)[i]=x[i−s] modulo n. Infinite grid: path on ℤ. For image I, AIB with A=Tᵣ, B=T[c]ᵀ gives I′[i,j]=I[i−r,j−c]. Shift matrices have one 1 per row/column and wrap-around; use identity for the unchanged axis. Zero padding breaks exact cyclic symmetry.|Periodisches 1D-Gitter = Zyklus. Translationen ℤₙ wirken durch (Tₛx)[i]=x[i−s] modulo n. Unendliches Gitter: Pfad auf ℤ. Für Bild I ergibt AIB mit A=Tᵣ, B=T[c]ᵀ den Shift I′[i,j]=I[i−r,j−c]. Shiftmatrizen haben eine 1 pro Zeile/Spalte und zyklischen Überlauf. Unveränderte Achse: Einheitsmatrix. Nullpadding bricht exakte zyklische Symmetrie.
LT₁=T₁L ⇔ L circulant ⇔ circular convolution by impulse response k: (k*x)[i]=Σⱼk[i−j]x[j]. Shift and circulants share DFT eigenvectors. FFT(x), FFT(k), multiply pointwise, inverse FFT → O(n log n).|LT₁=T₁L ⇔ L zirkulant ⇔ zyklische Faltung mit Impulsantwort k: (k*x)[i]=Σⱼk[i−j]x[j]. Shift und zirkulante Matrizen haben dieselben DFT-Eigenvektoren. FFT(x), FFT(k), punktweise multiplizieren, inverse FFT → O(n log n).
Space-frequency trade-off|Orts-Frequenz-Kompromiss
With e^(−iωt), standard deviations obey σₜσω≥1/2. Sketch with x-axis space and y-axis frequency: ordinary FT gives global horizontal frequency bands (no position). STFT Fourier-transforms local windows: uniform tiles, fixed window width, so narrow windows smear frequency. For x(at), spectrum is (1/∣a∣)x̂(ω/a): peaks move from ω₀ to aω₀. Tiny dilation shifts high peaks far in absolute frequency; even the translation-invariant modulus can change sharply.|Für e^(−iωt) gilt σₜσω≥1/2. Skizze mit Ort horizontal und Frequenz vertikal: FT ergibt globale horizontale Frequenzbänder ohne Ortsauflösung. STFT transformiert lokale Fenster: gleich große Kacheln, feste Fensterbreite. Schmale Fenster verschmieren Frequenzen. Für x(at) ist das Spektrum (1/∣a∣)x̂(ω/a): Spitzen wandern von ω₀ nach aω₀. Kleine Dilatation verschiebt hohe Spitzen absolut stark; selbst der translationsinvariante Betrag kann sich stark ändern.
Wavelets and scattering|Wavelets und Scattering
ψₐ,ᵦ(t)=a^(−1/2)ψ((t−b)/a), a>0. Translation shifts position and adds spectral phase; increasing a widens ψ in space and lowers/narrows its frequency band. Haar: adjacent +/− lobes detect a jump through local contrast. Sketch wavelet tiles: narrow in space/broad in frequency high up, wide in space/narrow in frequency low down.|ψₐ,ᵦ(t)=a^(−1/2)ψ((t−b)/a), a>0. Translation verschiebt den Ort und ändert die spektrale Phase. Größeres a streckt ψ im Ort und senkt/verengt sein Frequenzband. Haar: benachbarte +/−-Loben erkennen Sprünge durch lokalen Kontrast. Wavelet-Kacheln: oben schmal im Ort und breit in Frequenz, unten umgekehrt.
Scattering: cascade wavelet convolutions → modulus → optional further wavelet/modulus stages; average each path with low-pass φ[J] at scale 2ᴶ. Modulus demodulates oscillations into envelopes; direct averaging of zero-mean bands would suppress them. Sketch: wavelet bands become locally averaged envelopes with coarser spatial resolution. Gains local translation invariance at averaging scale and stability to small smooth deformations.|Scattering: Wavelet-Faltung → Betrag → optional weitere Wavelet-/Betragsstufen. Mittelung jedes Pfads mit Tiefpass φ_J auf Skala 2ᴶ. Betrag demoduliert Schwingungen zu Hüllkurven. Direkte Mittelung der mittelwertfreien Bänder würde sie unterdrücken. Skizze: lokal gemittelte Hüllkurven mit gröberer Ortsauflösung. Ergibt lokale Translationsinvarianz auf Mittelungsskala und Stabilität gegenüber kleinen glatten Deformationen.
@38-41|Group convolution|Gruppenfaltung
Transform and match|Transformieren und abgleichen
(x⋆ψ)(g)=∫Ω x(u) ψ(g⁻¹·u) dμ(u) for real signals and invariant μ. Output is indexed by g∈G. Next layer integrates over G with Haar measure and matches ψ(g⁻¹h). Equivariance: input rotation r yields y′(g)=y(r⁻¹g); response peak at g₀ moves to rg₀.|(x⋆ψ)(g)=∫Ω x(u) ψ(g⁻¹·u) dμ(u) für reelle Signale und invariantes μ. Ausgabe ist durch g∈G indiziert. Nächste Schicht integriert über G mit Haarmaß und gleicht ψ(g⁻¹h) ab. Äquivarianz: Eingaberotation r ergibt y′(g)=y(r⁻¹g). Antwortspitze bei g₀ wandert nach rg₀.
p4=ℤ²⋊C₄. First layer lifts image to a response at each location and each of four orientations. Later layers mix orientation channels via relative roto-translations; rotating the input rotates locations and cyclically shifts orientation channels.|p4=ℤ²⋊C₄. Erste Schicht hebt das Bild zu Antworten für jeden Ort und jede der vier Orientierungen an. Weitere Schichten mischen Orientierungskanäle über relative Rotationen und Translationen. Eingaberotation rotiert Orte und verschiebt Orientierungskanäle zyklisch.
Homogeneous space: transitive action (every u can be sent to every v). H=Stab(u₀)={g:g·u₀=u₀}; Ω≅G/H. Examples: sphere S²=SO(3)/SO(2) (rotations around pole); ℝ²=SE(2)/SO(2) (rotations about origin).|Homogener Raum: transitive Wirkung (jedes u lässt sich auf jedes v abbilden). H=Stab(u₀)={g:g·u₀=u₀}; Ω≅G/H. Beispiele: S²=SO(3)/SO(2) (Rotationen um Polachse); ℝ²=SE(2)/SO(2) (Rotationen um Ursprung).
||
@42-45|Manifolds|Mannigfaltigkeiten
Charts, tangent space and metric|Karten, Tangentialraum und Metrik
Manifolds model surfaces and low-dimensional data. A smooth d-manifold is locally parametrised by charts to open subsets of ℝᵈ, with smooth chart changes. TₚM contains tangent velocities at p. A Riemannian metric g gives a smoothly varying inner product on TₚM, hence lengths, angles and volume. Intrinsic quantities depend only on g, so bending that preserves g preserves them.|Mannigfaltigkeiten modellieren Oberflächen und niedrigdimensionale Daten. Eine glatte d-Mannigfaltigkeit besitzt Karten in offene Teilmengen von ℝᵈ mit glatten Kartenwechseln. TₚM enthält Tangentialgeschwindigkeiten in p. Eine riemannsche Metrik g gibt auf TₚM ein glatt variierendes Skalarprodukt und definiert Längen, Winkel und Volumen. Intrinsische Größen hängen nur von g ab und bleiben bei metrikerhaltender Biegung erhalten.
Geodesics: ∇_{γ′}γ′=0 (locally length-minimising for short segments). d_g(p,q)=inf lengths of curves from p to q. Both are intrinsic. A deformation is a diffeomorphism η of domains; exact isometry preserves metric (η*g′=g), equivalently geodesic distances for Riemannian isometries. Near-isometry means small metric/distance distortion. On homogeneous spaces compare to global group actions; on manifolds intrinsic stability uses metric distortion, since a transitive symmetry group need not exist.|Geodäten: ∇_{γ′}γ′=0 (kurze Abschnitte minimieren lokal die Länge). d_g(p,q)=Infimum der Kurvenlängen von p nach q. Beides ist intrinsisch. Eine Deformation ist ein Diffeomorphismus η zwischen Gebieten. Isometrien erhalten die Metrik (η*g′=g) bzw. geodätische Distanzen. Beinahe-Isometrien verzerren Metrik/Distanzen wenig. Homogene Räume: Vergleich mit globalen Gruppenwirkungen. Mannigfaltigkeiten: intrinsische Stabilität über metrische Verzerrung, auch ohne transitive Symmetriegruppe.
Differential operators|Differentialoperatoren
Gradient defined by ⟨grad f,v⟩_g=df(v): direction of steepest increase. Divergence measures local net outflow per volume. div=−grad* in L² under suitable boundary conditions. Convention L=−div grad≥0: measures deviation from local average, generates heat diffusion via ∂ₜu=−Lu. Intrinsic, linear, elliptic and self-adjoint with orthonormal eigenbasis on compact domains (appropriate boundary conditions).|Gradient durch ⟨grad f,v⟩_g=df(v) definiert: Richtung des steilsten Anstiegs. Divergenz misst lokalen Nettoausfluss pro Volumen. div=−grad* in L² bei passenden Randbedingungen. Konvention L=−div grad≥0: misst Abweichung vom lokalen Mittel und erzeugt Wärmediffusion über ∂ₜu=−Lu. Intrinsisch, linear, elliptisch und selbstadjungiert mit ON-Eigenbasis auf kompakten Gebieten (passende Randbedingungen).
@46-49|Spectral convolutions|Spektrale Faltungen
Fourier on a domain|Fourier-Analyse auf einem Gebiet
Eigenpairs Lφₖ=λₖφₖ. Transform x̂ₖ=⟨x,φₖ⟩; multiply by filter coefficient; reconstruct Σₖ h(λₖ)x̂ₖφₖ. Operator h(L) uses intrinsic L and commutes with transport by isometries. Free per-mode coefficients can depend on basis choices in repeated eigenspaces and react sharply to eigenvalue shifts/eigenvector rotations. Share one sufficiently smooth h(λ) across frequencies/domains, including repeated eigenvalues; smooth multipliers can improve localisation and stability.|Eigenpaare Lφₖ=λₖφₖ. Transformiere x̂ₖ=⟨x,φₖ⟩, multipliziere mit Filterkoeffizient, rekonstruiere Σₖ h(λₖ)x̂ₖφₖ. h(L) nutzt intrinsisches L und kommutiert mit Transport durch Isometrien. Freie Koeffizienten je Modus können bei mehrfachen Eigenwerten von der Basis abhängen und empfindlich auf verschobene Eigenwerte oder gedrehte Eigenvektoren reagieren. Eine hinreichend glatte Funktion h(λ) über Frequenzen/Gebiete teilen, einschließlich mehrfacher Eigenwerte. Das kann Lokalität und Stabilität verbessern.
Discrete geometry and message passing|Diskrete Geometrie und Message Passing
Triangulate a surface; edge lengths, angles and areas encode intrinsic metric. Cotangent stiffness and vertex masses give a mesh analogue of L (consistent with metric and area). Graph L=D−A: (Lx)ᵢ=ΣⱼAᵢⱼ(xᵢ−xⱼ), a shared, local, permutation-equivariant sum of differences. Mesh analogue weights neighbour differences geometrically and normalises by local area.|Oberfläche triangulieren. Kantenlängen, Winkel und Flächen kodieren die intrinsische Metrik. Kotangens-Gewichte und Knotenmassen liefern ein Mesh-Analogon von L (mit Metrik und Flächenmaß). Graph L=D−A: (Lx)ᵢ=ΣⱼAᵢⱼ(xᵢ−xⱼ), geteilte lokale permutationsäquivariante Differenzsumme. Im Mesh werden Differenzen geometrisch gewichtet und über lokale Flächen normiert.
Polynomial filters|Polynomfilter
Graph Fourier: L=UΛUᵀ, x̂=Uᵀx, x=Ux̂ (mass-weighted inner product for meshes). Full eigendecomposition is expensive; dense basis needs O(n²) storage and transform work. p(L)x=Σₖ₌₀ʳaₖLᵏx via r sparse multiplications, O(r(∣V∣+∣E∣)), support within r hops. Chebyshev recurrence is an alternative polynomial basis. Scalar spectral multipliers are isotropic (no orientation within a frequency), even if stable.|Graph-Fourier: L=UΛUᵀ, x̂=Uᵀx, x=Ux̂ (für Meshes massengewichtetes Skalarprodukt). Volle Eigenzerlegung ist teuer; dichte Basis benötigt O(n²) Speicher und Aufwand pro Transformation. p(L)x=Σₖ₌₀ʳaₖLᵏx über r dünnbesetzte Multiplikationen in O(r(∣V∣+∣E∣)), Reichweite r Hops. Chebyshev-Rekursion als alternative Polynombasis. Skalare Spektralfilter bleiben isotrop, auch bei Stabilität (keine Richtungsinformation innerhalb einer Frequenz)."""
for name, file in [('Body','arial.ttf'),('Bold','arialbd.ttf'),('Math','seguisym.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))

out = Path('output/pdf')
out.mkdir(parents=True, exist_ok=True)
W,H = A4
margin, gutter = 26, 17
colw = (W-2*margin-gutter)/2
top, bottom = H-75, 39
navy, teal, ink, grey = map(HexColor, ['#122739','#12655f','#17242d','#52626d'])

def safe_text(s, face):
    glyphs = pdfmetrics.getFont(face).face.charWidths
    fallback = pdfmetrics.getFont('Math').face.charWidths
    result = []
    for ch in s:
        if ord(ch) in glyphs:
            result.append(escape(ch))
        elif ord(ch) in fallback:
            result.append(f'<font name="Math">{escape(ch)}</font>')
        else:
            raise ValueError(f'Unsupported glyph {ch!r} U+{ord(ch):04X}')
    return ''.join(result)

def layout(raw, lang, size):
    items = []
    for line_no,line in enumerate(raw.strip().splitlines()):
        parts = line.split('|')
        if line.startswith('@'):
            label = f'{parts[0][1:]}   {parts[1+lang]}'
            kind, face, fs, lead, before, after, color = 'section','Bold',11,14, (9 if line_no else 0), 6,teal
        elif len(parts[lang]) < 62:
            label = parts[lang]
            kind, face, fs, lead, before, after, color = 'sub','Bold',size+0.25,size+2,2,1,navy
        else:
            label = parts[lang]
            kind, face, fs, lead, before, after, color = 'body','Body',size,size+2,0,4.1,ink
        style = ParagraphStyle(kind, fontName=face,fontSize=fs,leading=lead,textColor=color,splitLongWords=True)
        p = Paragraph(safe_text(label,face), style)
        _, h = p.wrap(colw, H)
        items.append((kind,p,h,before,after))
    return items

def diagrams(c, lang, pg, y):
    c.setStrokeColor(HexColor('#b2c5ca'))
    c.setFillColor(grey)
    c.setFont('Body',7)
    if pg == 0:
        x = margin+colw+gutter+19
        base = y-53
        c.line(x,base,x+colw-34,base)
        c.line(x,base,x,base+44)
        c.drawString(x+4,base+40,['Test error','Testfehler'][lang])
        c.drawRightString(x+colw-34,base-9,['Model complexity','Modellkomplexität'][lang])
        c.setDash(2,2)
        c.line(x+114,base,x+114,base+38)
        c.setDash()
        c.setStrokeColor(teal)
        p = c.beginPath()
        p.moveTo(x+9,base+31)
        p.curveTo(x+34,base+9,x+69,base+3,x+99,base+19)
        p.curveTo(x+108,base+24,x+110,base+35,x+114,base+36)
        p.curveTo(x+123,base+38,x+127,base+7,x+colw-44,base+7)
        c.drawPath(p)
        c.drawString(x+118,base+34,['Interpolation threshold','Interpolationsschwelle'][lang])
    else:
        c.drawString(margin,y-12,['Frequency vs. position','Frequenz über Ort'][lang])
        for i,title in enumerate(['Fourier','STFT','Wavelet']):
            x = margin+9+87*i
            base = y-68
            w,h = 68,44
            c.setStrokeColor(teal)
            c.setFillColor(HexColor('#edf5f4'))
            c.rect(x,base,w,h,fill=1)
            if i == 2:
                rows = [(13,2),(13,4),(18,8)]
            else:
                rows = [(11,1 if i==0 else 4)]*4
            for hh, n in rows:
                c.line(x,base,x+w,base)
                for j in range(1,n):
                    xx=x+w*j/n
                    c.line(xx,base,xx,base+hh)
                base += hh
            c.setFillColor(grey)
            c.drawCentredString(x+w/2,y-79,title)

for lang,code in enumerate(['EN','DE']):
    columns = data.split('\n||\n')
    for step in range(25):
        size = 10.2-step*0.10
        layouts = [layout(col,lang,size) for col in columns]
        heights = [sum(p[2]+p[3]+p[4] for p in l) for l in layouts]
        if max(heights) <= top-bottom:
            break
    else:
        raise ValueError(heights)
    print(code, size, heights)
    c = canvas.Canvas(str(out / f'Geometric_Techniques_Cheat_Sheet_{code}.pdf'), pagesize=A4)
    c.setTitle(f'Geometric Techniques | Cheat Sheet ({code})')
    for pg in range(2):
        c.setFillColor(navy)
        c.rect(0,H-6,W,6,fill=1,stroke=0)
        c.setFont('Bold',22)
        c.drawString(margin,H-34,'Geometric Techniques')
        c.setFillColor(teal)
        c.setFont('Bold',9)
        c.drawRightString(W-margin,H-32,['CHEAT SHEET  /  EN','SPICKZETTEL  /  DE'][lang])
        c.setFillColor(grey)
        c.setFont('Body',9)
        subtitle = [['Learning, symmetry, sets and graphs','Grids, group convolutions and manifolds'],['Lernen, Symmetrie, Mengen und Graphen','Gitter, Gruppenfaltungen und Mannigfaltigkeiten']][lang][pg]
        c.drawString(margin,H-51,subtitle)
        c.setStrokeColor(HexColor('#b2c5ca'))
        c.line(margin,H-59,W-margin,H-59)
        c.setStrokeColor(HexColor('#d3dfe0'))
        c.line(W/2,top,W/2,bottom)
        for col in range(2):
            x = margin+col*(colw+gutter)
            y = top
            for kind,p,h,before,after in layouts[2*pg+col]:
                y -= before+h
                p.drawOn(c,x,y)
                y -= after
            assert y >= bottom
            if (pg == 0 and col == 1) or (pg == 1 and col == 0):
                diagrams(c,lang,pg,y)
        c.setStrokeColor(HexColor('#b2c5ca'))
        c.line(margin,31,W-margin,31)
        c.setFillColor(grey)
        c.setFont('Body',7.5)
        source = ['Sources: Exam questions GDL and Oxford 2024 lecture slides', 'Quellen: Exam questions GDL und Oxford-2024-Vorlesungsfolien'][lang]
        c.drawString(margin,19,source)
        c.drawRightString(W-margin,19,f'{pg+1} / 2')
        c.showPage()
    c.save()

