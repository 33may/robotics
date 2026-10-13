-- Pandoc filter: drop LaTeXML wrapper noise so the markdown is plain.
function Div(el) return el.content end
function Span(el) return el.content end
function Header(el) el.attr = pandoc.Attr() return el end
function Link(el) el.attr = pandoc.Attr() return el end
function Image(el) el.attr = pandoc.Attr() return el end
function CodeBlock(el) el.attr = pandoc.Attr() return el end
function Code(el) el.attr = pandoc.Attr() return el end
function Table(el) el.attr = pandoc.Attr() return el end
function Figure(el)
  local blocks = {}
  for _, b in ipairs(el.content) do table.insert(blocks, b) end
  if el.caption and el.caption.long then
    for _, b in ipairs(el.caption.long) do table.insert(blocks, b) end
  end
  return blocks
end
