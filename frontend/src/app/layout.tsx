import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import "./globals.css";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-poppins",
});

export const metadata: Metadata = {
  title: "理想顔診断 - あなたの顔はどの国で魅力的？",
  description: "AIがあなたの顔を分析し、どの国の理想顔に最も近いかをランキング形式で診断します。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className={`${poppins.variable} font-sans antialiased text-slate-800`}>
        {children}
      </body>
    </html>
  );
}
