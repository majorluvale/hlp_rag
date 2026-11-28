import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Source } from '../api';

/**
 * Models often write `<br>` (mostly inside table cells). Raw HTML stays disabled for safety, so without this
 * the tags would show up as text: turn them into real Markdown line breaks, and drop any other stray tag.
 */
type MdNode = { type: string; value?: string; children?: MdNode[] };
const BR = /^<br\s*\/?>$/i;
function remarkBreaks() {
  const walk = (node: MdNode) => {
    if (!node.children) return;
    node.children = node.children.flatMap((c): MdNode[] => {
      if (c.type === 'html') {
        if (BR.test(c.value!.trim())) return [{ type: 'break' }];
        // a block like "a<br>b" arrives as one html node in some contexts: split it
        if (/<br\s*\/?>/i.test(c.value!)) {
          return c.value!.split(/<br\s*\/?>/i).flatMap((t, i) => (i ? [{ type: 'break' }, { type: 'text', value: t }] : [{ type: 'text', value: t }]));
        }
        return /^<\/?[a-z][^>]*>$/i.test(c.value!.trim()) ? [] : [{ type: 'text', value: c.value }];
      }
      walk(c);
      return [c];
    });
  };
  return (tree: MdNode) => walk(tree);
}

/** Renders the answer; turns [n] citations into clickable badges. */
export function Markdown({ text, sources = [], anchor = 'm' }: { text: string; sources?: Source[]; anchor?: string }) {
  const prepared = text.replace(/\[(\d{1,3})\](?!\()/g, '[$1](#cite-$1)');
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith('#cite-')) {
              const n = Number(href.slice(6));
              const s = sources.find((x) => x.n === n);
              const link = s?.url ?? undefined;
              return (
                <a className="cite" href={link ?? `#${anchor}-src-${n}`} target={link ? '_blank' : undefined} rel="noreferrer" title={s ? s.title : `Source ${n}`}
                  onClick={(e) => {
                    if (!link) {
                      e.preventDefault();
                      document.getElementById(`${anchor}-src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                  }}>
                  {n}
                </a>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {prepared}
      </ReactMarkdown>
    </div>
  );
}
