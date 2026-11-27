import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Source } from '../api';

/** Renders the answer; turns [n] citations into clickable badges. */
export function Markdown({ text, sources = [], anchor = 'm' }: { text: string; sources?: Source[]; anchor?: string }) {
  const prepared = text.replace(/\[(\d{1,3})\](?!\()/g, '[$1](#cite-$1)');
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
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
