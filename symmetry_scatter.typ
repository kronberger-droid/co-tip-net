#import "@preview/lilaq:0.5.0" as lq

#set page(width: 16cm, height: 12cm, margin: 1.5cm)

#let data = csv("symmetry_results.csv")

#let goods = data.filter(row => row.at(0) == "1")
#let bads = data.filter(row => row.at(0) == "0")

#let good_x = goods.map(row => float(row.at(1)))
#let good_y = goods.map(row => float(row.at(2)))
#let bad_x = bads.map(row => float(row.at(1)))
#let bad_y = bads.map(row => float(row.at(2)))

#lq.diagram(
  xlabel: [Inertia eigenvalue ratio],
  ylabel: [Rotational autocorrelation],
  title: [Symmetry metrics: good vs bad CO tips],
  legend: (position: bottom + right),
  lq.scatter(
    bad_x,
    bad_y,
    mark: "o",
    color: red.transparentize(50%),
    stroke: none,
    label: [bad],
  ),
  lq.scatter(
    good_x,
    good_y,
    mark: "o",
    color: blue.transparentize(50%),
    stroke: none,
    label: [good],
  ),
)
